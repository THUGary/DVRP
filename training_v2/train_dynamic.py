"""
Training script for Dynamic VRP Model (Static + Adapter).

Two training modes:
1. Supervised Learning: Use optimal/heuristic solutions as labels
2. Reinforcement Learning: REINFORCE with environment rewards

The static model is frozen, only adapters are trained.

NORMALIZATION (v2 - capacity-normalized):
- Grid coordinates: [0, grid_size] => [0, 1] by dividing by COORD_NORM
- Demands: demand / DEMAND_NORM (where DEMAND_NORM = vehicle capacity = 30)
- Loads: load / capacity
- Time: time / max_time
"""

from __future__ import annotations
import os
import argparse
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import DEMAND_NORM, COORD_NORM  # Standardized normalization constants
from models_v2.dynamic_model import DynamicVRPModel, create_dynamic_model
from training_v2.credit_assignment import (
    CreditAssigner, 
    CoordinationLoss, 
    compute_step_credits,
    compute_step_credits_with_balance,
    enhanced_reinforce_loss,
    WorkloadBalanceLoss,
    WorkloadTracker,
    compute_balance_bonus,
)


class SimpleDVRPEnv:
    """
    Simple DVRP environment for training.
    
    Dynamic demands arrive over time, agents must visit them before deadlines.
    
    Uses standardized normalization constants from configs.py:
    - capacity: DEMAND_NORM = 30 (vehicle capacity)
    - grid_size: COORD_NORM (default 20)
    """
    
    def __init__(
        self,
        grid_size: int = int(COORD_NORM),
        num_agents: int = 2,
        capacity: float = DEMAND_NORM,  # Use standardized constant (= 30)
        max_time: int = 100,
        num_demands: int = 20,
        device: str = "cpu",
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.capacity = capacity
        self.max_time = max_time
        self.num_demands = num_demands
        self.device = torch.device(device)
        
        # State
        self.time: int = 0
        self.depot: torch.Tensor = None
        self.demands: List[Dict] = []  # list of demand dicts
        self.agent_positions: torch.Tensor = None
        self.agent_loads: torch.Tensor = None
        self.agent_targets: List[Optional[int]] = None
        self.served: set = set()
        
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset environment."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.time = 0
        
        # Random depot
        self.depot = torch.randint(0, self.grid_size, (2,), device=self.device).float()
        
        # Generate all demands upfront (static-like for simplicity)
        self.demands = []
        total_cap = 0
        max_cap_per_demand = self.capacity // self.num_agents
        
        for i in range(self.num_demands):
            demand = {
                'id': i,
                'x': torch.randint(0, self.grid_size, (1,), device=self.device).item(),
                'y': torch.randint(0, self.grid_size, (1,), device=self.device).item(),
                'capacity': min(np.random.randint(1, 10), max_cap_per_demand),
                'release_time': 0,  # all at start for static
                'deadline': self.max_time * 2,
            }
            self.demands.append(demand)
            total_cap += demand['capacity']
        
        # Initialize agents at depot
        self.agent_positions = self.depot.unsqueeze(0).expand(self.num_agents, -1).clone()
        self.agent_loads = torch.zeros(self.num_agents, device=self.device)
        self.agent_targets = [None] * self.num_agents
        self.served = set()
        
        return self._get_obs()
    
    def _get_obs(self) -> Dict:
        """Get current observation."""
        # Active demands (released and not served)
        active_demands = [
            d for d in self.demands 
            if d['release_time'] <= self.time and d['id'] not in self.served
        ]
        
        # Node features: (x, y, capacity, deadline)
        if active_demands:
            node_xy = torch.tensor(
                [[d['x'], d['y']] for d in active_demands],
                dtype=torch.float32, device=self.device
            )
            node_demand = torch.tensor(
                [d['capacity'] for d in active_demands],
                dtype=torch.float32, device=self.device
            )
            node_deadline = torch.tensor(
                [d['deadline'] for d in active_demands],
                dtype=torch.float32, device=self.device
            )
            node_ids = [d['id'] for d in active_demands]
        else:
            node_xy = torch.zeros(1, 2, device=self.device)
            node_demand = torch.zeros(1, device=self.device)
            node_deadline = torch.ones(1, device=self.device) * self.max_time * 2
            node_ids = []
        
        # Agent states: (x, y, load, time)
        agent_states = torch.cat([
            self.agent_positions,
            self.agent_loads.unsqueeze(-1),
            torch.full((self.num_agents, 1), self.time, device=self.device),
        ], dim=-1)
        
        return {
            'depot_xy': self.depot.unsqueeze(0).unsqueeze(0),  # (1, 1, 2)
            'node_xy': node_xy.unsqueeze(0),  # (1, n_nodes, 2)
            'node_demand': node_demand.unsqueeze(0),  # (1, n_nodes)
            'node_deadline': node_deadline.unsqueeze(0),  # (1, n_nodes)
            'node_ids': node_ids,
            'agent_states': agent_states.unsqueeze(0),  # (1, n_agents, 4)
            'time': self.time,
        }
    
    def step(self, actions: List[int]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute actions.
        
        Args:
            actions: list of target node indices per agent (0=depot, 1..N=nodes)
            
        Returns:
            obs, reward, done, info
        """
        obs = self._get_obs()
        node_ids = obs['node_ids']
        
        reward = 0.0
        info = {'served': [], 'distances': []}
        
        for agent_idx, action in enumerate(actions):
            if action == 0:
                # Go to depot
                target = self.depot
                # Refill at depot
                self.agent_loads[agent_idx] = 0
            elif 1 <= action <= len(node_ids):
                # Go to node
                demand_id = node_ids[action - 1]
                demand = self.demands[demand_id]
                target = torch.tensor([demand['x'], demand['y']], 
                                       dtype=torch.float32, device=self.device)
                
                # Check capacity
                if self.agent_loads[agent_idx] + demand['capacity'] <= self.capacity:
                    # Serve demand
                    self.served.add(demand_id)
                    self.agent_loads[agent_idx] += demand['capacity']
                    reward += 1.0  # bonus for serving
                    info['served'].append(demand_id)
            else:
                # Invalid action, stay in place
                target = self.agent_positions[agent_idx]
            
            # Move agent (simplified: teleport)
            distance = torch.norm(target - self.agent_positions[agent_idx]).item()
            self.agent_positions[agent_idx] = target
            reward -= distance * 0.01  # distance penalty
            info['distances'].append(distance)
        
        self.time += 1
        
        # Check termination
        all_served = len(self.served) == self.num_demands
        timeout = self.time >= self.max_time
        done = all_served or timeout
        
        if done and not all_served:
            # Penalty for unserved demands
            reward -= (self.num_demands - len(self.served)) * 5.0
        
        return self._get_obs(), reward, done, info
    
    def get_mask(self, obs: Dict) -> torch.Tensor:
        """
        Get action mask.
        
        Args:
            obs: current observation
            
        Returns:
            mask: (1, n_agents, n_nodes+1) -inf for invalid, 0 for valid
        """
        n_nodes = obs['node_demand'].size(1)
        n_agents = self.num_agents
        
        mask = torch.zeros(1, n_agents, n_nodes + 1, device=self.device)
        
        # Check capacity for each agent-node pair
        for agent_idx in range(n_agents):
            current_load = self.agent_loads[agent_idx].item()
            for node_idx in range(n_nodes):
                demand = obs['node_demand'][0, node_idx].item()
                if current_load + demand > self.capacity:
                    mask[0, agent_idx, node_idx + 1] = float('-inf')
        
        return mask


def train_rl_epoch(
    model: DynamicVRPModel,
    optimizer: torch.optim.Optimizer,
    env: SimpleDVRPEnv,
    n_episodes: int,
    device: torch.device,
    use_credit_assignment: bool = True,
    use_balance_training: bool = True,
    balance_weight: float = 0.5,
) -> Tuple[float, float, Dict]:
    """
    Train one epoch with RL and Multi-Agent Credit Assignment.
    
    Args:
        model: DynamicVRPModel to train
        optimizer: Optimizer
        env: Training environment
        n_episodes: Number of episodes per epoch
        device: Device
        use_credit_assignment: Whether to use credit assignment
        use_balance_training: Whether to add workload balance rewards
        balance_weight: Weight for balance component in reward
        
    Returns:
        avg_reward: Average episode reward
        avg_loss: Average loss
        balance_metrics: Dict with balance statistics
    """
    model.train()
    
    total_reward = 0.0
    total_loss = 0.0
    
    # Balance metrics aggregation
    all_balance_scores = []
    all_distance_cvs = []
    all_node_count_cvs = []
    
    # Initialize credit assignment modules
    credit_assigner = CreditAssigner(
        num_agents=env.num_agents,
        coordination_bonus=2.0,
        collision_penalty=5.0,
        distance_factor=0.01,
        coverage_bonus=1.0,
    )
    coordination_loss_fn = CoordinationLoss(penalty_weight=0.5)
    workload_balance_loss_fn = WorkloadBalanceLoss(
        distance_weight=1.0,
        node_count_weight=0.5,
        demand_weight=0.5,
    ) if use_balance_training else None
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        
        episode_reward = 0.0
        log_probs = []
        probs_list = []
        global_rewards = []
        individual_rewards_list = []
        
        # Initialize workload tracker for balance training
        workload_tracker = WorkloadTracker(env.num_agents, device) if use_balance_training else None
        
        done = False
        while not done:
            # Get mask
            mask = env.get_mask(obs)
            
            # Normalize inputs
            depot_norm = obs['depot_xy'] / env.grid_size
            node_norm = obs['node_xy'] / env.grid_size
            demand_norm = obs['node_demand'] / env.capacity
            agent_states = obs['agent_states'].clone()
            agent_states[:, :, :2] /= env.grid_size
            agent_states[:, :, 2] /= env.capacity
            agent_states[:, :, 3] /= env.max_time
            
            # Forward with full probs for coordination loss
            selected, probs = model.forward_with_full_probs(
                depot_xy=depot_norm,
                node_xy=node_norm,
                node_demand=demand_norm,
                agent_states=agent_states,
                ninf_mask=mask,
                node_deadline=obs['node_deadline'],
                time_now=torch.tensor([obs['time']], device=device, dtype=torch.float32),
            )
            
            # Get log probs
            selected_probs = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1)
            log_probs.append(selected_probs.log().sum())
            probs_list.append(probs)
            
            # Step environment
            actions = selected[0].tolist()
            obs_next, reward, done, info = env.step(actions)
            
            # Update workload tracker
            if workload_tracker is not None:
                # Extract demand served per agent
                demands_served = {}
                for i, action in enumerate(actions):
                    if action != 0:  # Not depot
                        node_idx = action - 1
                        if node_idx < obs['node_demand'].size(1):
                            demands_served[i] = obs['node_demand'][0, node_idx].item()
                
                workload_tracker.update(
                    actions=actions,
                    distances=info.get('distances', [0.0] * env.num_agents),
                    demands_served=demands_served,
                )
            
            # Compute individual credits
            if use_credit_assignment:
                # Build target positions tensor
                depot_pos = env.depot
                node_positions = obs['node_xy'][0]  # (n_nodes, 2)
                target_positions = torch.cat([
                    depot_pos.unsqueeze(0),  # (1, 2)
                    node_positions * env.grid_size,  # (n_nodes, 2) - unnormalize
                ], dim=0)
                
                if use_balance_training and workload_tracker is not None:
                    # Use balance-aware credit assignment
                    individual_rewards = compute_step_credits_with_balance(
                        credit_assigner=credit_assigner,
                        workload_tracker=workload_tracker,
                        actions=actions,
                        env_info=info,
                        global_reward=reward,
                        agent_positions=env.agent_positions,
                        target_positions=target_positions,
                        balance_weight=balance_weight,
                    )
                else:
                    individual_rewards = compute_step_credits(
                        credit_assigner=credit_assigner,
                        actions=actions,
                        env_info=info,
                        global_reward=reward,
                        agent_positions=env.agent_positions,
                        target_positions=target_positions,
                    )
                individual_rewards_list.append(individual_rewards)
            else:
                # Equal split of global reward
                individual_rewards_list.append(
                    torch.full((env.num_agents,), reward / env.num_agents, device=device)
                )
            
            global_rewards.append(reward)
            episode_reward += reward
            obs = obs_next
        
        # Add episode-end balance bonus
        if use_balance_training and workload_tracker is not None:
            balance_bonus = workload_tracker.get_balance_reward(scale=2.0)
            episode_reward += balance_bonus
            global_rewards[-1] += balance_bonus  # Add to last step reward
            
            # Collect balance metrics
            metrics = workload_tracker.get_balance_metrics()
            all_balance_scores.append(metrics['balance_score'])
            all_distance_cvs.append(metrics['distance_cv'])
            all_node_count_cvs.append(metrics['node_count_cv'])
        
        # Compute loss with credit assignment
        if log_probs:
            if use_credit_assignment:
                loss, loss_info = enhanced_reinforce_loss(
                    log_probs=log_probs,
                    probs_list=probs_list,
                    global_rewards=global_rewards,
                    credit_assigner=credit_assigner,
                    coordination_loss_fn=coordination_loss_fn,
                    individual_rewards_list=individual_rewards_list,
                )
                
                # Add workload balance loss
                if use_balance_training and workload_balance_loss_fn is not None:
                    # Aggregate workload for the episode
                    agent_dists = workload_tracker.total_distances.unsqueeze(0)  # (1, agents)
                    agent_counts = workload_tracker.node_counts.unsqueeze(0)  # (1, agents)
                    agent_demands = workload_tracker.demand_served.unsqueeze(0)  # (1, agents)
                    
                    balance_loss, _ = workload_balance_loss_fn(
                        agent_dists, agent_counts, agent_demands
                    )
                    loss = loss + balance_weight * balance_loss
            else:
                # Original REINFORCE
                log_prob_sum = torch.stack(log_probs).sum()
                loss = -episode_reward * log_prob_sum
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        total_reward += episode_reward
    
    # Aggregate balance metrics
    balance_metrics = {}
    if use_balance_training and all_balance_scores:
        balance_metrics = {
            'mean_balance_score': np.mean(all_balance_scores),
            'mean_distance_cv': np.mean(all_distance_cvs),
            'mean_node_count_cv': np.mean(all_node_count_cvs),
        }
    
    return total_reward / n_episodes, total_loss / n_episodes, balance_metrics


def train_supervised_epoch(
    model: DynamicVRPModel,
    optimizer: torch.optim.Optimizer,
    env: SimpleDVRPEnv,
    n_episodes: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one epoch with supervised learning (greedy labels)."""
    model.train()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        
        done = False
        while not done:
            mask = env.get_mask(obs)
            
            # Normalize inputs
            depot_norm = obs['depot_xy'] / env.grid_size
            node_norm = obs['node_xy'] / env.grid_size
            demand_norm = obs['node_demand'] / env.capacity
            agent_states = obs['agent_states'].clone()
            agent_states[:, :, :2] /= env.grid_size
            agent_states[:, :, 2] /= env.capacity
            agent_states[:, :, 3] /= env.max_time
            
            # Get greedy labels (closest valid node)
            n_nodes = obs['node_xy'].size(1)
            labels = []
            for agent_idx in range(env.num_agents):
                agent_pos = env.agent_positions[agent_idx]
                best_action = 0  # depot by default
                best_dist = torch.norm(agent_pos - env.depot).item()
                
                for node_idx in range(n_nodes):
                    if mask[0, agent_idx, node_idx + 1] == 0:  # valid
                        node_pos = obs['node_xy'][0, node_idx]
                        dist = torch.norm(agent_pos - node_pos * env.grid_size).item()
                        if dist < best_dist:
                            best_dist = dist
                            best_action = node_idx + 1
                
                labels.append(best_action)
            
            labels = torch.tensor(labels, device=device).unsqueeze(0)
            
            # Forward
            model.encode(
                depot_xy=depot_norm,
                node_xy=node_norm,
                node_demand=demand_norm,
                node_deadline=obs['node_deadline'],
                time_now=torch.tensor([obs['time']], device=device, dtype=torch.float32),
            )
            
            # Get logits
            agent_emb = model.agent_embed(agent_states)
            agent_emb = model.context_adapter(agent_emb)
            load_ratio = agent_states[:, :, 2]
            
            # Manual decoding for logits
            input_cat = torch.cat((agent_emb, load_ratio.unsqueeze(-1)), dim=-1)
            q_last = model.static_model.decoder._reshape_by_heads(
                model.static_model.decoder.Wq_last(input_cat)
            )
            attn = model.static_model.decoder._multi_head_attention(
                q_last, 
                model.static_model.decoder.k, 
                model.static_model.decoder.v, 
                mask
            )
            mh_out = model.static_model.decoder.multi_head_combine(attn)
            logits = torch.matmul(mh_out, model.static_model.decoder.single_head_key)
            logits = logits / model.static_model.decoder.sqrt_embedding_dim
            logits = logits + mask
            
            # Loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            pred = logits.argmax(dim=-1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.numel()
            
            # Step with greedy actions
            obs, _, done, _ = env.step(labels[0].tolist())
    
    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(n_episodes, 1)
    
    return accuracy, avg_loss


def train_dynamic_model(
    static_checkpoint: str,
    mode: str = "rl",  # "rl" or "supervised"
    grid_size: int = 20,
    num_agents: int = 2,
    num_demands: int = 20,
    adapter_dim: int = 32,
    epochs: int = 100,
    episodes_per_epoch: int = 100,
    lr: float = 1e-4,
    save_dir: str = "checkpoints/dynamic_vrp",
    device: str = "cuda",
    use_balance_training: bool = True,
    balance_weight: float = 0.5,
):
    """
    Train dynamic VRP model with multi-vehicle balance.
    
    Args:
        static_checkpoint: path to pretrained static model
        mode: training mode ("rl" or "supervised")
        grid_size: environment grid size
        num_agents: number of agents
        num_demands: number of demands per episode
        adapter_dim: adapter dimension
        epochs: number of epochs
        episodes_per_epoch: episodes per epoch
        lr: learning rate
        save_dir: save directory
        device: device
        use_balance_training: enable workload balance training
        balance_weight: weight for balance component
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training Dynamic VRP Model on {device}")
    print(f"Mode: {mode}")
    print(f"Balance Training: {use_balance_training} (weight={balance_weight})")
    
    # Create model
    model = create_dynamic_model(
        static_model_or_checkpoint=static_checkpoint,
        adapter_dim=adapter_dim,
        freeze_static=True,
        device=str(device),
    )
    
    # Create environment
    env = SimpleDVRPEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_demands=num_demands,
        device=device,
    )
    
    # Optimizer (only adapter params)
    optimizer = Adam(model.get_trainable_params(), lr=lr)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_metric = float('-inf') if mode == "rl" else 0.0
    
    for epoch in range(1, epochs + 1):
        if mode == "rl":
            reward, loss, balance_metrics = train_rl_epoch(
                model, optimizer, env, episodes_per_epoch, device,
                use_credit_assignment=True,
                use_balance_training=use_balance_training,
                balance_weight=balance_weight,
            )
            
            # Print with balance metrics
            if balance_metrics:
                print(f"Epoch {epoch}/{epochs}: Reward={reward:.4f}, Loss={loss:.4f}, "
                      f"Balance={balance_metrics['mean_balance_score']:.3f}, "
                      f"DistCV={balance_metrics['mean_distance_cv']:.3f}")
            else:
                print(f"Epoch {epoch}/{epochs}: Reward={reward:.4f}, Loss={loss:.4f}")
            metric = reward
        else:
            accuracy, loss = train_supervised_epoch(model, optimizer, env, episodes_per_epoch, device)
            print(f"Epoch {epoch}/{epochs}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
            metric = accuracy
        
        # Save best
        if metric > best_metric:
            best_metric = metric
            torch.save({
                'adapter_state_dict': model.adapter_state_dict(),
                'epoch': epoch,
                'metric': metric,
            }, os.path.join(save_dir, f"best_adapter_{mode}.pt"))
            print(f"  New best: {best_metric:.4f}")
        
        # Periodic save
        if epoch % 10 == 0:
            torch.save({
                'adapter_state_dict': model.adapter_state_dict(),
                'epoch': epoch,
                'metric': metric,
            }, os.path.join(save_dir, f"adapter_{mode}_ep{epoch}.pt"))
    
    print(f"\nTraining complete. Best metric: {best_metric:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Dynamic VRP Model")
    parser.add_argument("--static-checkpoint", type=str, required=True,
                        help="Path to pretrained static model")
    parser.add_argument("--mode", type=str, default="rl", choices=["rl", "supervised"])
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--num-demands", type=int, default=20)
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints/dynamic_vrp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-balance-training", action="store_true", default=True,
                        help="Enable multi-vehicle workload balance training")
    parser.add_argument("--no-balance-training", action="store_false", dest="use_balance_training",
                        help="Disable balance training")
    parser.add_argument("--balance-weight", type=float, default=0.5,
                        help="Weight for balance component in reward")
    
    args = parser.parse_args()
    
    train_dynamic_model(
        static_checkpoint=args.static_checkpoint,
        mode=args.mode,
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_demands=args.num_demands,
        adapter_dim=args.adapter_dim,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        use_balance_training=args.use_balance_training,
        balance_weight=args.balance_weight,
    )


if __name__ == "__main__":
    main()
