"""
Simple evaluation script for VRP models.

TERMINOLOGY:
- num_nodes: Actual number of customer/demand nodes (used for tensor shapes)
- total_demand: Upper limit of sum of all demands (NOT used in this file)
"""

from __future__ import annotations
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_v2.static_model import StaticVRPModel, create_static_model


def evaluate_static_model(
    checkpoint: str,
    num_nodes: int = 20,
    num_instances: int = 100,
    pomo_size: int = 20,
    num_vehicles: int = 2,
    augment: bool = False,
    device: str = "cuda",
):
    """
    Evaluate static VRP model.
    
    Args:
        checkpoint: path to model checkpoint
        num_nodes: Number of customer nodes (distinct from total_demand which is capacity upper limit)
        num_instances: number of test instances
        pomo_size: number of POMO rollouts
        num_vehicles: number of vehicles
        augment: use 8-fold augmentation
        device: device
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_static_model(embedding_dim=128, encoder_layers=6, heads=8)
    
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {checkpoint}")
    
    model = model.to(device)
    model.eval()
    
    # Generate test instances
    print(f"\nEvaluating on {num_instances} random instances...")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  POMO size: {pomo_size}")
    print(f"  Vehicles: {num_vehicles}")
    print(f"  Augmentation: {augment}")
    
    total_distance = 0.0
    
    for i in range(num_instances):
        # Generate random instance
        depot_xy = torch.rand(1, 1, 2, device=device)
        node_xy = torch.rand(1, num_nodes, 2, device=device)
        
        if num_nodes == 20:
            demand_scaler = 30
        elif num_nodes == 50:
            demand_scaler = 40
        else:
            demand_scaler = 50
        
        node_demand = torch.randint(1, 10, (1, num_nodes), device=device).float() / demand_scaler
        
        # Solve
        distances, routes = model.solve(
            depot_xy=depot_xy,
            node_xy=node_xy,
            node_demand=node_demand,
            pomo_size=pomo_size,
            num_vehicles=num_vehicles,
            augment=augment,
        )
        
        total_distance += distances[0].item()
        
        if (i + 1) % 10 == 0:
            print(f"  Instance {i+1}/{num_instances}: dist={distances[0].item():.4f}")
    
    avg_distance = total_distance / num_instances
    print(f"\nAverage distance: {avg_distance:.4f}")
    
    return avg_distance


def main():
    parser = argparse.ArgumentParser(description="Evaluate VRP Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-nodes", type=int, default=20,
                        help="Number of customer nodes (NOT total_demand which is capacity upper limit)")
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--pomo-size", type=int, default=20)
    parser.add_argument("--num-vehicles", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    evaluate_static_model(
        checkpoint=args.checkpoint,
        num_nodes=args.num_nodes,
        num_instances=args.num_instances,
        pomo_size=args.pomo_size,
        num_vehicles=args.num_vehicles,
        augment=args.augment,
        device=args.device,
    )


if __name__ == "__main__":
    main()
