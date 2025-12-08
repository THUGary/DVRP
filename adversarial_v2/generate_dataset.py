"""
Dataset Generation for Static VRP Training

This module generates training and test datasets for static VRP model training.
It can generate problems using either:
1. Diffusion model (conditional generation)
2. Random generation (uniform distribution)

Output format matches train_static.py expectations:
- depot_xy: (num_problems, 1, 2) - depot coordinates
- node_xy: (num_problems, num_nodes, 2) - customer coordinates
- node_demand: (num_problems, num_nodes) - normalized demands

All coordinates are in [0, 1] range (normalized).
Demands are normalized by vehicle capacity (default: demands/30).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import prepare_condition, CONDITION_DIM

# Default constants (matching train_static.py)
DEFAULT_NUM_NODES = 20
DEFAULT_MAX_DEMAND = 5
DEMAND_NORM = 30.0  # Vehicle capacity for normalization


def generate_random_problems(
    num_problems: int,
    num_nodes: int,
    device: torch.device,
    target_num_vehicles: int = 4,
    max_c: int = DEFAULT_MAX_DEMAND,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random VRP problems (uniform distribution).
    
    Args:
        num_problems: Number of problems to generate
        num_nodes: Number of customer nodes per problem
        device: Torch device
        target_num_vehicles: Target number of vehicles (controls total demand)
    
    Returns:
        depot_xy: (num_problems, 1, 2)
        node_xy: (num_problems, num_nodes, 2)
        node_demand: (num_problems, num_nodes) - normalized
    """
    # Random depot and node coordinates in [0, 1]
    depot_xy = torch.rand(num_problems, 1, 2, device=device)
    node_xy = torch.rand(num_problems, num_nodes, 2, device=device)
    
    # Generate demands: [1, MAX_DEMAND] then normalize
    raw_demand = torch.randint(
        1, max_c + 1, 
        (num_problems, num_nodes), 
        device=device, 
        dtype=torch.float
    )
    
    # Scale demands to match target vehicles
    # Average total demand ≈ target_num_vehicles * capacity
    avg_demand = (1 + max_c) / 2.0  # Expected single node demand
    expected_total = avg_demand * num_nodes
    target_total = target_num_vehicles  # In normalized units (capacity = 1.0)
    
    # Scale factor
    scale = target_total / (expected_total / DEMAND_NORM)
    raw_demand = (raw_demand * scale).clamp(1, max_c)
    
    # Normalize by capacity
    node_demand = raw_demand / DEMAND_NORM
    
    return depot_xy, node_xy, node_demand


def generate_diffusion_problems(
    model_path: str,
    num_problems: int,
    num_nodes: int,
    device: torch.device,
    total_demand: int = 60,
    max_c: int = DEFAULT_MAX_DEMAND,
    batch_size: int = 100,
    use_ddim: bool = True,
    ddim_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate VRP problems using diffusion model.
    
    The diffusion model generates demands in a different format (time, x, y, capacity, lifetime).
    We convert this to static VRP format by:
    1. Ignoring time and lifetime (static setting)
    2. Using x, y as node coordinates
    3. Using capacity as demand
    
    Args:
        model_path: Path to diffusion model checkpoint
        num_problems: Number of problems to generate
        num_nodes: Number of customer nodes per problem
        device: Torch device
        total_demand: Total demand for conditional generation
        max_c: Maximum demand per node
        batch_size: Batch size for generation
        use_ddim: Use DDIM sampling (faster)
        ddim_steps: Number of DDIM steps
    
    Returns:
        depot_xy: (num_problems, 1, 2)
        node_xy: (num_problems, num_nodes, 2)
        node_demand: (num_problems, num_nodes) - normalized
    """
    # Load diffusion model
    model = DemandDiffusionModel(condition_dim=CONDITION_DIM)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both state_dict and full checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Prepare condition with only total_demand and max_c (other params use defaults)
    condition = prepare_condition(total_demand=total_demand, max_c=max_c).unsqueeze(0).to(device)
    
    # Generate problems in batches
    all_depot_xy = []
    all_node_xy = []
    all_node_demand = []
    
    num_batches = (num_problems + batch_size - 1) // batch_size
    
    print(f"Generating {num_problems} problems with diffusion model...")
    print(f"  Model: {model_path}")
    print(f"  Condition: total_demand={total_demand}, max_c={max_c}")
    print(f"  Batch size: {batch_size}, Num batches: {num_batches}")
    print(f"  DDIM: {use_ddim}, Steps: {ddim_steps if use_ddim else 1000}")
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_problems - batch_idx * batch_size)
            
            batch_depot_xy = []
            batch_node_xy = []
            batch_node_demand = []
            
            for _ in range(current_batch_size):
                # Generate demands using diffusion model
                # Output shape: (num_nodes, 5) - [time, x, y, capacity, lifetime]
                if use_ddim:
                    generated = model.sample_ddim(
                        condition, num_nodes, 
                        grid_size=(100, 100),  # Not used but required
                        num_inference_steps=ddim_steps
                    )
                else:
                    generated = model.sample(
                        condition, num_nodes,
                        grid_size=(100, 100)
                    )
                
                # Extract coordinates and demands
                # Diffusion output is normalized [0, 1] already
                node_x = generated[:, 1].clamp(0, 1)  # x coordinate
                node_y = generated[:, 2].clamp(0, 1)  # y coordinate
                node_c = generated[:, 3].clamp(0, 1)  # capacity (normalized)
                
                # Stack coordinates
                node_xy = torch.stack([node_x, node_y], dim=-1)  # (num_nodes, 2)
                
                # Convert demand back to [1, max_c] then re-normalize
                raw_demand = node_c * max_c
                raw_demand = raw_demand.clamp(1, max_c)
                node_demand = raw_demand / DEMAND_NORM
                
                # Random depot (separate from nodes)
                depot_xy = torch.rand(1, 2, device=device)
                
                batch_depot_xy.append(depot_xy)
                batch_node_xy.append(node_xy)
                batch_node_demand.append(node_demand)
            
            # Stack batch
            all_depot_xy.append(torch.stack(batch_depot_xy))
            all_node_xy.append(torch.stack(batch_node_xy))
            all_node_demand.append(torch.stack(batch_node_demand))
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  Generated {min((batch_idx + 1) * batch_size, num_problems)}/{num_problems} problems")
    
    # Concatenate all batches
    depot_xy = torch.cat(all_depot_xy, dim=0)
    node_xy = torch.cat(all_node_xy, dim=0)
    node_demand = torch.cat(all_node_demand, dim=0)
    
    return depot_xy, node_xy, node_demand


def save_dataset(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save dataset to file.
    
    Args:
        depot_xy: (num_problems, 1, 2)
        node_xy: (num_problems, num_nodes, 2)
        node_demand: (num_problems, num_nodes)
        output_path: Output file path (.pt)
        metadata: Optional metadata to save
    """
    data = {
        'depot_xy': depot_xy.cpu(),
        'node_xy': node_xy.cpu(),
        'node_demand': node_demand.cpu(),
        'num_problems': depot_xy.size(0),
        'num_nodes': node_xy.size(1),
    }
    
    if metadata:
        data['metadata'] = metadata
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    torch.save(data, output_path)
    print(f"Saved dataset to {output_path}")
    print(f"  Problems: {data['num_problems']}")
    print(f"  Nodes: {data['num_nodes']}")


def load_dataset(
    input_path: str,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load dataset from file.
    
    Args:
        input_path: Input file path (.pt)
        device: Device to load data to
    
    Returns:
        depot_xy: (num_problems, 1, 2)
        node_xy: (num_problems, num_nodes, 2)
        node_demand: (num_problems, num_nodes)
    """
    data = torch.load(input_path, map_location=device)
    
    return (
        data['depot_xy'].to(device),
        data['node_xy'].to(device),
        data['node_demand'].to(device),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate training and test datasets for Static VRP"
    )
    
    # Generation mode
    parser.add_argument(
        "--mode", type=str, default="random",
        choices=["random", "diffusion"],
        help="Generation mode: random (uniform) or diffusion (model-based)"
    )
    
    # Diffusion model settings
    parser.add_argument(
        "--diffusion-checkpoint", type=str, default=None,
        help="Path to diffusion model checkpoint (required for diffusion mode)"
    )
    parser.add_argument(
        "--use-ddim", action="store_true", default=True,
        help="Use DDIM sampling for faster generation"
    )
    parser.add_argument(
        "--ddim-steps", type=int, default=50,
        help="Number of DDIM sampling steps"
    )
    
    # Dataset settings
    parser.add_argument(
        "--total-episodes", type=int, default=10000,
        help="Total number of problems to generate"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Ratio of problems for test set (0.0 to 1.0)"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=DEFAULT_NUM_NODES,
        help="Number of customer nodes per problem"
    )
    parser.add_argument(
        "--target-vehicles", type=int, default=4,
        help="Target number of vehicles (controls demand scaling)"
    )
    parser.add_argument(
        "--total-demand", type=int, default=60,
        help="Target total demand (sum of node capacities) per problem before normalization"
    )
    parser.add_argument(
        "--max-c", type=int, default=DEFAULT_MAX_DEMAND,
        help="Maximum demand per node (integer >=1)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", type=str, default="data/static_vrp",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--train-file", type=str, default="train.pt",
        help="Training dataset filename"
    )
    parser.add_argument(
        "--test-file", type=str, default="test.pt",
        help="Test dataset filename"
    )
    
    # Other settings
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size for generation (diffusion mode)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate split sizes
    num_test = int(args.total_episodes * args.test_ratio)
    num_train = args.total_episodes - num_test
    
    print(f"\n{'='*60}")
    print(f"Dataset Generation Configuration")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Total episodes: {args.total_episodes}")
    print(f"Train/Test split: {num_train}/{num_test} ({1-args.test_ratio:.0%}/{args.test_ratio:.0%})")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Target vehicles: {args.target_vehicles}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Validate diffusion mode
    if args.mode == "diffusion" and args.diffusion_checkpoint is None:
        parser.error("--diffusion-checkpoint is required for diffusion mode")
    
    # Generate training data
    print(f"Generating {num_train} training problems...")
    if args.mode == "random":
        train_depot, train_nodes, train_demand = generate_random_problems(
            num_train, args.num_nodes, device, args.target_vehicles, max_c=args.max_c
        )
    else:
        train_depot, train_nodes, train_demand = generate_diffusion_problems(
            args.diffusion_checkpoint, num_train, args.num_nodes, device,
            total_demand=args.total_demand, max_c=args.max_c,
            batch_size=args.batch_size, use_ddim=args.use_ddim, ddim_steps=args.ddim_steps
        )
    
    # Generate test data
    print(f"\nGenerating {num_test} test problems...")
    if args.mode == "random":
        test_depot, test_nodes, test_demand = generate_random_problems(
            num_test, args.num_nodes, device, args.target_vehicles, max_c=args.max_c
        )
    else:
        test_depot, test_nodes, test_demand = generate_diffusion_problems(
            args.diffusion_checkpoint, num_test, args.num_nodes, device,
            total_demand=args.total_demand, max_c=args.max_c,
            batch_size=args.batch_size, use_ddim=args.use_ddim, ddim_steps=args.ddim_steps
        )
    
    # Prepare metadata
    metadata = {
        'mode': args.mode,
        'num_nodes': args.num_nodes,
        'target_vehicles': args.target_vehicles,
        'seed': args.seed,
        'diffusion_checkpoint': args.diffusion_checkpoint if args.mode == "diffusion" else None,
    }
    
    # Save datasets
    train_path = os.path.join(args.output_dir, args.train_file)
    test_path = os.path.join(args.output_dir, args.test_file)
    
    print(f"\nSaving datasets...")
    save_dataset(train_depot, train_nodes, train_demand, train_path, metadata)
    save_dataset(test_depot, test_nodes, test_demand, test_path, metadata)
    
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"{'='*60}")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")


if __name__ == "__main__":
    main()
