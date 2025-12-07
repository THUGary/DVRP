"""
Training script for Static VRP Model (POMO-style).

POMO training:
1. Generate random VRP instances
2. Run multiple parallel rollouts (pomo_size) with diverse starting points
3. Use self-competitive baseline: advantage = reward - mean(rewards)
4. REINFORCE gradient update

NORMALIZATION SCHEME (v2 - capacity-normalized):
------------------------------------------------
Training uses a normalized coordinate space [0,1] x [0,1] with vehicle_capacity=1.0.
- Demands: random [1,5] / 30 = [0.033, 0.167] of capacity
- Vehicle capacity: 1.0 (represents capacity=30 in real terms)

At inference time (V2Planner), we map from the environment space to model space:
- Coordinates: coords / COORD_NORM (maps grid to [0,1])
- Demands: demand / DEMAND_NORM (where DEMAND_NORM = 30 = vehicle capacity)
- Vehicle Capacity: capacity / DEMAND_NORM = 30/30 = 1.0

KEY DESIGN:
- capacity=30 and max_demand=5 are FIXED
- Map size (COORD_NORM) can be varied: 20, 30, 40, etc.

TERMINOLOGY:
- num_nodes: Number of customer/demand nodes (for tensor shapes)
- total_demand: Upper limit of sum of all customer demands (NOT node count!)
"""

from __future__ import annotations
import os
import argparse
from datetime import datetime
from typing import Optional, Tuple, Iterator
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_v2.static_model import StaticVRPModel, StaticVRPEnv, create_static_model

# Default max demand per node (demands are 1 to MAX_DEMAND_PER_NODE)
MAX_DEMAND_PER_NODE = 5
AVG_DEMAND_PER_NODE = (1 + MAX_DEMAND_PER_NODE) / 2  # = 3.0

# Default number of nodes
DEFAULT_NUM_NODES = 50


class DatasetLoader:
    """
    Iterator for loading pre-generated VRP problems from file.
    
    Supports infinite iteration with optional shuffling.
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
    ):
        """
        Args:
            data_path: Path to .pt file with pre-generated problems
            batch_size: Batch size for training
            device: Device to load data to
            shuffle: Whether to shuffle data each epoch
        """
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        
        # Load data
        print(f"Loading dataset from {data_path}...")
        data = torch.load(data_path, map_location='cpu')
        
        self.depot_xy = data['depot_xy']  # (num_problems, 1, 2)
        self.node_xy = data['node_xy']    # (num_problems, num_nodes, 2)
        self.node_demand = data['node_demand']  # (num_problems, num_nodes)
        
        self.num_problems = self.depot_xy.size(0)
        self.num_nodes = self.node_xy.size(1)
        self.num_batches = self.num_problems // batch_size
        
        print(f"  Loaded {self.num_problems} problems with {self.num_nodes} nodes")
        print(f"  Batches per epoch: {self.num_batches}")
        
        self._indices = torch.arange(self.num_problems)
        self._current_idx = 0
    
    def _shuffle_data(self):
        """Shuffle indices for new epoch."""
        if self.shuffle:
            self._indices = torch.randperm(self.num_problems)
        self._current_idx = 0
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get next batch of problems.
        
        Returns:
            depot_xy: (batch_size, 1, 2)
            node_xy: (batch_size, num_nodes, 2)
            node_demand: (batch_size, num_nodes)
        """
        # Check if we need to start new epoch
        if self._current_idx + self.batch_size > self.num_problems:
            self._shuffle_data()
        
        # Get batch indices
        batch_indices = self._indices[self._current_idx:self._current_idx + self.batch_size]
        self._current_idx += self.batch_size
        
        # Get batch data
        depot_xy = self.depot_xy[batch_indices].to(self.device)
        node_xy = self.node_xy[batch_indices].to(self.device)
        node_demand = self.node_demand[batch_indices].to(self.device)
        
        return depot_xy, node_xy, node_demand
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.num_batches


def generate_random_problems(
    batch_size: int,
    num_nodes: int,
    device: torch.device,
    target_num_vehicles: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random VRP instances.
    
    Args:
        batch_size: number of instances
        num_nodes: number of customer nodes
        device: torch device
        target_num_vehicles: target number of vehicles needed (controls total demand)
        
    Returns:
        depot_xy: (batch, 1, 2)
        node_xy: (batch, num_nodes, 2)
        node_demand: (batch, num_nodes) - normalized so vehicle_capacity=1.0
        
    Note:
        Map size: [0,1] x [0,1] (unit square)
        Vehicle capacity: 1.0 (represents capacity=30 in real terms)
        Per-node demand: random in [1,5] / 30, so [0.033, 0.167]
        Total demand is scaled to need ~target_num_vehicles trips
    """
    depot_xy = torch.rand(batch_size, 1, 2, device=device)
    node_xy = torch.rand(batch_size, num_nodes, 2, device=device)
    
    # Generate raw demands [1, 5] (matching MAX_DEMAND_PER_NODE=5)
    # mean = 3.0
    raw_demand = torch.randint(1, MAX_DEMAND_PER_NODE + 1, (batch_size, num_nodes), device=device).float()
    
    # Normalize: demand / 30 (matching DEMAND_NORM=30)
    # So demands are in [0.033, 0.167] range, vehicle_capacity = 1.0
    node_demand = raw_demand / 30.0
    
    # Note: With num_nodes=20, mean demand=3/30=0.1, total demand ≈ 20*0.1 = 2.0
    # This means we need ~2 vehicle trips on average for 20 nodes
    # This is reasonable for target_num_vehicles=2
    
    return depot_xy, node_xy, node_demand


def train_one_batch(
    model: StaticVRPModel,
    env: StaticVRPEnv,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_nodes: int,
    pomo_size: int,
    aug_factor: int,
    device: torch.device,
    target_num_vehicles: int = 4,
    preloaded_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[float, float]:
    """
    Train on one batch.
    
    Args:
        preloaded_data: Optional tuple (depot_xy, node_xy, node_demand) for pre-generated data.
                        If None, will generate random problems.
    
    Returns:
        avg_score: average tour length (lower is better)
        avg_loss: average policy gradient loss
    """
    model.train()
    
    # Use preloaded data or generate random problems
    if preloaded_data is not None:
        depot_xy, node_xy, node_demand = preloaded_data
    else:
        depot_xy, node_xy, node_demand = generate_random_problems(
            batch_size, num_nodes, device, target_num_vehicles
        )
    env.load_problems(depot_xy, node_xy, node_demand, aug_factor=aug_factor)
    
    # Reset
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)
    
    # Collect rollout
    prob_list = []
    state, _, done = env.pre_step()
    
    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        prob_list.append(prob)
    
    # Stack probabilities
    prob_tensor = torch.stack(prob_list, dim=2)  # (batch, pomo, steps)
    
    # Compute loss (REINFORCE with POMO baseline)
    # reward shape: (batch, pomo)
    advantage = reward - reward.mean(dim=1, keepdim=True)  # POMO baseline
    log_prob = prob_tensor.log().sum(dim=2)  # (batch, pomo)
    
    loss = -(advantage * log_prob).mean()
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Score: best tour length across POMO instances
    best_reward, _ = reward.max(dim=1)
    avg_score = -best_reward.mean().item()
    
    return avg_score, loss.item()


def train_static_model(
    num_nodes: int = DEFAULT_NUM_NODES,
    pomo_size: int = 100,
    aug_factor: int = 1,
    embedding_dim: int = 128,
    encoder_layers: int = 6,
    heads: int = 8,
    epochs: int = 100,
    episodes_per_epoch: int = 10000,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    save_dir: str = "checkpoints/static_vrp",
    save_interval: int = 10,
    device: str = "cuda",
    resume_from: Optional[str] = None,
    target_num_vehicles: int = 4,
    patience: int = 20,
    threshold: float = 1e-4,
    train_data: Optional[str] = None,
):
    """
    Train static VRP model.
    
    Args:
        num_nodes: Number of customer/demand nodes
        pomo_size: number of parallel rollouts
        embedding_dim: model dimension
        encoder_layers: number of encoder layers
        heads: number of attention heads
        epochs: number of training epochs
        episodes_per_epoch: training episodes per epoch (ignored when train_data is set)
        batch_size: batch size
        lr: learning rate
        weight_decay: weight decay
        save_dir: directory to save checkpoints
        save_interval: save every N epochs
        device: cuda or cpu
        resume_from: checkpoint to resume from
        target_num_vehicles: target number of vehicles
        patience: early stopping patience
        threshold: early stopping threshold
        train_data: path to pre-generated training data (.pt file)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Load dataset if provided
    data_loader = None
    if train_data and os.path.exists(train_data):
        data_loader = DatasetLoader(train_data, batch_size, device, shuffle=True)
        num_nodes = data_loader.num_nodes  # Use num_nodes from dataset
        print(f"Using pre-generated dataset: {train_data}")
    elif train_data:
        print(f"WARNING: train_data file not found: {train_data}, using random generation")
    
    # Create model
    model = create_static_model(
        embedding_dim=embedding_dim,
        encoder_layers=encoder_layers,
        heads=heads,
    ).to(device)
    
    # Create environment (uses num_nodes as problem_size)
    env = StaticVRPEnv(problem_size=num_nodes, pomo_size=pomo_size)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.8), int(epochs * 0.95)], gamma=0.1)
    
    # Resume
    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from {resume_from}, epoch {start_epoch}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine number of batches per epoch
    if data_loader is not None:
        n_batches = len(data_loader)
        data_source = "pre-generated dataset"
    else:
        n_batches = episodes_per_epoch // batch_size
        data_source = "random generation"
    
    # Training loop
    print(f"\nTraining Static VRP Model")
    print(f"  Data source: {data_source}")
    print(f"  Num nodes: {num_nodes} (demand node count)")
    print(f"  Target vehicles: {target_num_vehicles}")
    print(f"  Map size: [0,1] x [0,1] (unit square)")
    print(f"  Vehicle capacity: 1.0")
    print(f"  Avg total demand: ~{target_num_vehicles:.1f}")
    print(f"  POMO size: {pomo_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batches/epoch: {n_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Early stopping: patience={patience}, threshold={threshold}")
    print()
    
    best_score = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_score = 0.0
        epoch_loss = 0.0
        
        # Shuffle data at start of each epoch (for dataset loader)
        if data_loader is not None:
            data_loader._shuffle_data()
        
        for batch_idx in range(n_batches):
            # Get data (from loader or generate)
            if data_loader is not None:
                preloaded_data = data_loader.get_batch()
            else:
                preloaded_data = None
            
            score, loss = train_one_batch(
                model, env, optimizer, batch_size, num_nodes, pomo_size, aug_factor, device,
                target_num_vehicles=target_num_vehicles,
                preloaded_data=preloaded_data,
            )
            epoch_score += score
            epoch_loss += loss
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx+1}/{n_batches}: "
                      f"Score={score:.4f}, Loss={loss:.4f}")
        
        scheduler.step()
        
        avg_score = epoch_score / n_batches
        avg_loss = epoch_loss / n_batches
        
        print(f"Epoch {epoch}/{epochs}: Avg Score={avg_score:.4f}, Avg Loss={avg_loss:.4f}")
        
        # Save checkpoint only for: best score OR final epoch
        # (Avoid saving every save_interval to prevent checkpoint bloat)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'score': avg_score,
            'num_nodes': num_nodes,
        }
        
        if avg_score < best_score - threshold:
            best_score = avg_score
            no_improve_epochs = 0
            torch.save(checkpoint, os.path.join(save_dir, f"best_n{num_nodes}.pt"))
            print(f"  New best score: {best_score:.4f}")
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs}/{patience} epochs (best: {best_score:.4f})")
            if no_improve_epochs >= patience:
                print(f"  Early stopping triggered after {epoch} epochs.")
                torch.save(checkpoint, os.path.join(save_dir, f"final_n{num_nodes}_ep{epoch}.pt"))
                print(f"  Saved final checkpoint at epoch {epoch}")
                break
        
        # Save final epoch checkpoint for resuming
        if epoch == epochs:
            torch.save(checkpoint, os.path.join(save_dir, f"final_n{num_nodes}_ep{epoch}.pt"))
            print(f"  Saved final checkpoint at epoch {epoch}")
    
    print(f"\nTraining complete. Best score: {best_score:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Static VRP Model")
    parser.add_argument("--num-nodes", type=int, default=DEFAULT_NUM_NODES,
                        help="Number of demand nodes")
    parser.add_argument("--target-vehicles", type=int, default=4,
                        help="Target number of vehicles (controls total demand)")
    parser.add_argument("--pomo-size", type=int, default=20)
    parser.add_argument("--aug-factor", type=int, default=1, choices=[1, 8],
                        help="Augmentation factor to use when loading problems (1 or 8). Using 8 will expand each batch by 8x.")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints/static_vrp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Early stopping threshold")
    parser.add_argument("--train-data", type=str, default=None,
                        help="Path to pre-generated training data (.pt file). If provided, uses this instead of random generation.")
    
    args = parser.parse_args()
    
    train_static_model(
        num_nodes=args.num_nodes,
        pomo_size=args.pomo_size,
        aug_factor=args.aug_factor,
        embedding_dim=args.embedding_dim,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        resume_from=args.resume,
        target_num_vehicles=args.target_vehicles,
        patience=args.patience,
        threshold=args.threshold,
        train_data=args.train_data,
    )


if __name__ == "__main__":
    main()
