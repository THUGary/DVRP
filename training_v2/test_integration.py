"""
Integration test - use new models with existing DVRP environment.
"""

from __future__ import annotations
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import DEMAND_NORM, COORD_NORM
from models_v2.static_model import StaticVRPModel, create_static_model
from models_v2.dynamic_model import DynamicVRPModel, create_dynamic_model
from environment.env_tensor import TensorGridEnvironment, TensorEnvObservation


class SimpleStaticGenerator:
    """Simple static demand generator for testing."""
    
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
        
        # Generate random demands
        for i in range(self.num_demands):
            x = self._rng.randint(0, self.width - 1)
            y = self._rng.randint(0, self.height - 1)
            c = self._rng.randint(10, self.max_capacity)
            
            # Create demand tuple (compatible with environment)
            from agent.generator.base import Demand
            demand = Demand(
                x=x, y=y, t=0, c=c, end_t=1000, service_time=0
            )
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
) -> dict:
    """
    Convert environment observation to model input format.
    
    Args:
        obs: TensorEnvObservation from environment
        env_width: environment width for normalization
        env_height: environment height for normalization
        capacity: vehicle capacity for demand normalization
        
    Returns:
        dict with model input tensors
    """
    batch_size = obs.time.size(0)
    device = obs.time.device
    
    # Depot position (normalized to [0,1])
    depot_xy = obs.depot.float().unsqueeze(1)  # (batch, 1, 2)
    depot_xy[..., 0] /= env_width
    depot_xy[..., 1] /= env_height
    
    # Node positions (demands that are active)
    active_mask = obs.active_mask()  # (batch, max_demands)
    
    # Normalize positions
    node_xy = obs.demands_pos.float()  # (batch, max_demands, 2)
    node_xy[..., 0] /= env_width
    node_xy[..., 1] /= env_height
    
    # Normalize demands
    node_demand = obs.demands_capacity.float() / capacity  # (batch, max_demands)
    
    # Agent states: (x, y, load, time)
    agent_pos_norm = obs.agent_pos.float()
    agent_pos_norm[..., 0] /= env_width
    agent_pos_norm[..., 1] /= env_height
    
    agent_load_norm = obs.agent_load.float() / capacity
    agent_time = obs.time.float().unsqueeze(1).expand(-1, obs.agent_pos.size(1))
    agent_time_norm = agent_time / 100.0  # Assume max_time=100
    
    agent_states = torch.cat([
        agent_pos_norm,
        agent_load_norm.unsqueeze(-1),
        agent_time_norm.unsqueeze(-1),
    ], dim=-1)  # (batch, n_agents, 4)
    
    # Create mask (inf for inactive demands, 0 for active)
    ninf_mask = torch.zeros_like(active_mask, dtype=torch.float32)
    ninf_mask[~active_mask] = float('-inf')
    
    # Add depot column (depot always allowed)
    depot_mask = torch.zeros(batch_size, 1, device=device)
    ninf_mask = torch.cat([depot_mask, ninf_mask], dim=1)  # (batch, max_demands+1)
    
    # Expand for agents
    n_agents = obs.agent_pos.size(1)
    ninf_mask = ninf_mask.unsqueeze(1).expand(-1, n_agents, -1)  # (batch, n_agents, max_demands+1)
    
    return {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
        'agent_states': agent_states,
        'ninf_mask': ninf_mask,
        'active_mask': active_mask,
    }


def model_output_to_action(
    selected: torch.Tensor,
    obs: TensorEnvObservation,
    env_width: int = 20,
    env_height: int = 20,
) -> torch.Tensor:
    """
    Convert model output (node indices) to environment actions (movement directions).
    
    The environment expects actions in [-1, 0, 1] representing movement direction,
    not absolute target coordinates.
    
    Args:
        selected: (batch, n_agents) node indices (0=depot, 1..N=demands)
        obs: environment observation
        env_width, env_height: environment dimensions
        
    Returns:
        actions: (batch, n_agents, 2) movement directions in {-1, 0, 1}
    """
    batch_size, n_agents = selected.shape
    device = selected.device
    
    # Prepare position lookup: depot + demand positions
    depot_pos = obs.depot.unsqueeze(1)  # (batch, 1, 2)
    demand_pos = obs.demands_pos  # (batch, max_demands, 2)
    all_pos = torch.cat([depot_pos, demand_pos], dim=1)  # (batch, max_demands+1, 2)
    
    # Get target positions and compute movement direction
    actions = torch.zeros(batch_size, n_agents, 2, dtype=torch.long, device=device)
    for b in range(batch_size):
        for a in range(n_agents):
            node_idx = selected[b, a].item()
            target = all_pos[b, node_idx]  # (2,)
            current = obs.agent_pos[b, a]  # (2,)
            
            # Compute direction: sign of (target - current), clamped to [-1, 1]
            delta = target - current
            direction = delta.sign().clamp(-1, 1)
            actions[b, a] = direction
    
    return actions


def run_episode_with_model(
    env: TensorGridEnvironment,
    model: DynamicVRPModel,
    max_steps: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Run one episode using the dynamic model for decisions.
    
    Returns:
        dict with episode statistics
    """
    device = next(model.parameters()).device
    
    obs = env.reset()
    total_reward = 0.0
    total_served = 0.0
    steps = 0
    
    for step in range(max_steps):
        # Convert observation to model input
        model_input = obs_to_model_input(
            obs,
            env_width=env.width,
            env_height=env.height,
            capacity=env.capacity,
        )
        
        # Move to device
        for k, v in model_input.items():
            model_input[k] = v.to(device)
        
        # Get model prediction
        with torch.no_grad():
            selected, probs = model(
                depot_xy=model_input['depot_xy'],
                node_xy=model_input['node_xy'],
                node_demand=model_input['node_demand'],
                agent_states=model_input['agent_states'],
                ninf_mask=model_input['ninf_mask'],
            )
        
        # Convert to environment action
        actions = model_output_to_action(
            selected.cpu(),
            obs,
            env_width=env.width,
            env_height=env.height,
        )
        
        # Step environment
        obs, reward, done, info = env.step(actions)
        total_reward += reward.sum().item()
        total_served += info.get('service_bonus', torch.zeros(1)).sum().item()
        steps += 1
        
        if verbose and step % 10 == 0:
            active = obs.active_mask().sum(dim=1).float().mean().item()
            print(f"  Step {step}: reward={reward.mean().item():.3f}, active_demands={active:.1f}")
        
        if done.all():
            break
    
    return {
        'total_reward': total_reward,
        'total_served': total_served,
        'steps': steps,
    }


def main():
    """Run integration test."""
    print("Integration Test: New Models + Existing Environment")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create models
    print("\nCreating models...")
    static_model = create_static_model(
        embedding_dim=128,
        encoder_layers=6,
        heads=8,
    )
    
    dynamic_model = create_dynamic_model(
        static_model,
        adapter_dim=32,
        freeze_static=True,
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in dynamic_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in dynamic_model.parameters() if p.requires_grad):,}")
    
    # Create environment
    print("\nCreating environment...")
    generator = SimpleStaticGenerator(
        width=int(COORD_NORM),
        height=int(COORD_NORM),
        num_demands=20,
        max_capacity=50,
        seed=42,
    )
    
    env = TensorGridEnvironment(
        width=int(COORD_NORM),
        height=int(COORD_NORM),
        num_agents=2,
        capacity=int(DEMAND_NORM),
        depot=(10, 10),
        batch_size=4,
        max_demands=64,
        generator=generator,
        device="cpu",
        max_time=100,
    )
    
    print(f"  Grid: {env.width}x{env.height}")
    print(f"  Agents: {env.num_agents}")
    print(f"  Capacity: {env.capacity}")
    print(f"  Batch size: {env.batch_size}")
    
    # Run episode
    print("\nRunning episode with model...")
    results = run_episode_with_model(env, dynamic_model, max_steps=100, verbose=True)
    
    print(f"\nResults:")
    print(f"  Total reward: {results['total_reward']:.2f}")
    print(f"  Total served: {results['total_served']:.2f}")
    print(f"  Steps: {results['steps']}")
    
    print("\n" + "=" * 60)
    print("Integration test completed!")


if __name__ == "__main__":
    main()
