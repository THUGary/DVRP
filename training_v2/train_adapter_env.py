"""
Train dynamic adapter using real DVRP environment.
"""

from __future__ import annotations
import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict, List
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import DEMAND_NORM, COORD_NORM
from models_v2.static_model import create_static_model
from models_v2.dynamic_model import create_dynamic_model, DynamicVRPModel
from environment.env_tensor import TensorGridEnvironment, TensorEnvObservation
from agent.generator.base import Demand


class StaticDemandGenerator:
    """Generate static demands at t=0."""
    
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        num_demands: int = 20,
        max_capacity: int = 50,
        seed: int = None,
    ):
        self.width = width
        self.height = height
        self.num_demands = num_demands
        self.max_capacity = max_capacity
        self.params = {}
        self._rng = None
        self._demands = []
        self._released = False
        if seed is not None:
            self.reset(seed)
    
    def reset(self, seed=None):
        import random
        self._rng = random.Random(seed)
        self._demands = []
        self._released = False
        
        for _ in range(self.num_demands):
            x = self._rng.randint(0, self.width - 1)
            y = self._rng.randint(0, self.height - 1)
            c = self._rng.randint(10, self.max_capacity)
            demand = Demand(x=x, y=y, t=0, c=c, end_t=1000, service_time=0)
            self._demands.append(demand)
    
    def sample(self, t):
        if t == 0 and not self._released:
            self._released = True
            return list(self._demands)
        return []


def obs_to_model_input(
    obs: TensorEnvObservation,
    env_width: float = COORD_NORM,
    env_height: float = COORD_NORM,
    capacity: float = DEMAND_NORM,
    max_time: float = 100.0,
) -> Dict[str, torch.Tensor]:
    """Convert environment observation to model input."""
    batch_size = obs.time.size(0)
    device = obs.time.device
    
    # Depot
    depot_xy = obs.depot.float().unsqueeze(1) / env_width
    
    # Nodes
    node_xy = obs.demands_pos.float() / env_width
    node_demand = obs.demands_capacity.float() / capacity
    
    # Agent states
    agent_pos_norm = obs.agent_pos.float() / env_width
    agent_load_norm = obs.agent_load.float() / capacity
    agent_time_norm = obs.time.float().unsqueeze(1).expand(-1, obs.agent_pos.size(1)) / max_time
    
    agent_states = torch.cat([
        agent_pos_norm,
        agent_load_norm.unsqueeze(-1),
        agent_time_norm.unsqueeze(-1),
    ], dim=-1)
    
    # Mask
    active_mask = obs.active_mask()
    n_agents = obs.agent_pos.size(1)
    ninf_mask = torch.zeros(batch_size, n_agents, obs.demands_pos.size(1) + 1, device=device)
    
    # Mask inactive demands
    for b in range(batch_size):
        for a in range(n_agents):
            ninf_mask[b, a, 1:][~active_mask[b]] = float('-inf')
    
    return {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
        'agent_states': agent_states,
        'ninf_mask': ninf_mask,
        'active_mask': active_mask,
    }


def model_to_action(
    selected: torch.Tensor,
    obs: TensorEnvObservation,
) -> torch.Tensor:
    """Convert model output to environment action (movement direction)."""
    batch_size, n_agents = selected.shape
    device = selected.device
    
    depot_pos = obs.depot.unsqueeze(1)
    demand_pos = obs.demands_pos
    all_pos = torch.cat([depot_pos, demand_pos], dim=1)
    
    actions = torch.zeros(batch_size, n_agents, 2, dtype=torch.long, device=device)
    for b in range(batch_size):
        for a in range(n_agents):
            node_idx = selected[b, a].item()
            target = all_pos[b, node_idx]
            current = obs.agent_pos[b, a]
            delta = target - current
            actions[b, a] = delta.sign().clamp(-1, 1)
    
    return actions


def rollout_episode(
    env: TensorGridEnvironment,
    model: DynamicVRPModel,
    max_steps: int = 100,
    device: str = "cuda",
) -> Dict:
    """Run one episode and collect trajectories."""
    obs = env.reset()
    
    # First step to spawn demands
    actions = torch.zeros(env.batch_size, env.num_agents, 2, dtype=torch.long, device=env.device)
    obs, _, _, _ = env.step(actions)
    
    log_probs_list = []
    rewards_list = []
    
    total_reward = 0.0
    
    for step in range(max_steps):
        model_input = obs_to_model_input(
            obs,
            env_width=env.width,
            env_height=env.height,
            capacity=env.capacity,
            max_time=env.max_time,
        )
        
        # Move to model device
        for k, v in model_input.items():
            model_input[k] = v.to(device)
        
        # Forward
        selected, probs = model(
            depot_xy=model_input['depot_xy'],
            node_xy=model_input['node_xy'],
            node_demand=model_input['node_demand'],
            agent_states=model_input['agent_states'],
            ninf_mask=model_input['ninf_mask'],
        )
        
        # Compute log probs (probs are already softmax outputs from decode_step)
        log_probs = torch.log(probs + 1e-8)  # (batch, n_agents)
        log_probs_list.append(log_probs.sum(dim=1))  # sum over agents
        
        # Step environment
        actions = model_to_action(selected.cpu(), obs)
        obs, reward, done, info = env.step(actions)
        
        rewards_list.append(reward)
        total_reward += reward.sum().item()
        
        if done.all():
            break
    
    # Stack trajectories
    log_probs = torch.stack(log_probs_list, dim=1)  # (batch, steps)
    rewards = torch.stack(rewards_list, dim=1)  # (batch, steps)
    
    return {
        'log_probs': log_probs,
        'rewards': rewards,
        'total_reward': total_reward,
        'steps': len(rewards_list),
        'served': env.stats['served_count'].sum().item(),
    }


def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns."""
    batch_size, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(batch_size, device=rewards.device)
    
    for t in reversed(range(T)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
    
    return returns


def train_adapter(
    static_checkpoint: str,
    num_episodes: int = 1000,
    batch_size: int = 16,
    num_demands: int = 20,
    lr: float = 1e-4,
    gamma: float = 0.99,
    device: str = "cuda",
    save_dir: str = "checkpoints/dynamic_adapter",
    log_interval: int = 50,
):
    """Train dynamic adapter with REINFORCE."""
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Create models
    static_model = create_static_model(embedding_dim=128, encoder_layers=6, heads=8)
    ckpt = torch.load(static_checkpoint, map_location=device)
    static_model.load_state_dict(ckpt['model_state_dict'])
    
    dynamic_model = create_dynamic_model(
        static_model,
        adapter_dim=32,
        freeze_static=True,
    ).to(device)
    
    print(f"Trainable parameters: {sum(p.numel() for p in dynamic_model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    optimizer = Adam(dynamic_model.get_trainable_params(), lr=lr)
    
    # Environment
    generator = StaticDemandGenerator(
        width=20,
        height=20,
        num_demands=num_demands,
        max_capacity=50,
    )
    
    env = TensorGridEnvironment(
        width=int(COORD_NORM),
        height=int(COORD_NORM),
        num_agents=2,
        capacity=int(DEMAND_NORM),
        depot=(10, 10),
        batch_size=batch_size,
        max_demands=64,
        generator=generator,
        device="cpu",
        max_time=100,
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Training
    best_served = 0
    reward_history = []
    
    for episode in range(num_episodes):
        # Reseed generator for variety
        generator.reset(seed=episode)
        
        # Rollout
        dynamic_model.train()
        result = rollout_episode(env, dynamic_model, max_steps=100, device=device)
        
        # Compute returns
        returns = compute_returns(result['rewards'].to(device), gamma=gamma)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # REINFORCE loss
        loss = -(result['log_probs'] * returns).mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dynamic_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        reward_history.append(result['total_reward'])
        
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(reward_history[-log_interval:])
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={result['total_reward']:.2f}, "
                  f"Served={result['served']:.0f}, "
                  f"AvgReward={avg_reward:.2f}")
        
        # Save best
        if result['served'] > best_served:
            best_served = result['served']
            torch.save({
                'episode': episode,
                'adapter_state': dynamic_model.adapter_state_dict(),
                'served': best_served,
            }, os.path.join(save_dir, 'best_adapter.pt'))
            print(f"  New best served: {best_served}")
    
    # Save final
    torch.save({
        'episode': num_episodes,
        'adapter_state': dynamic_model.adapter_state_dict(),
    }, os.path.join(save_dir, 'final_adapter.pt'))
    
    print(f"\nTraining complete. Best served: {best_served}")
    return dynamic_model


def main():
    parser = argparse.ArgumentParser(description="Train Dynamic Adapter")
    parser.add_argument("--static-checkpoint", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-demands", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints/dynamic_adapter")
    
    args = parser.parse_args()
    
    train_adapter(
        static_checkpoint=args.static_checkpoint,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        num_demands=args.num_demands,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
