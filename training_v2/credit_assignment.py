"""
Multi-Agent Credit Assignment for Dynamic VRP Training.

This module implements various credit assignment mechanisms to improve
multi-vehicle coordination in RL training:

1. Individual Credit: Each agent receives reward based on its own contribution
2. Difference Reward: Counterfactual evaluation (what if agent wasn't there)
3. Coordination Bonus: Extra reward for good division of labor
4. Collision Penalty: Penalty when multiple agents target same node
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class CreditAssigner:
    """
    Multi-Agent Credit Assignment mechanism.
    
    Decomposes global reward into individual agent contributions
    to improve learning signal for coordination.
    """
    
    def __init__(
        self,
        num_agents: int,
        coordination_bonus: float = 2.0,
        collision_penalty: float = 5.0,
        distance_factor: float = 0.01,
        coverage_bonus: float = 1.0,
        use_difference_reward: bool = True,
    ):
        """
        Args:
            num_agents: Number of agents in the system
            coordination_bonus: Bonus for agents selecting different targets
            collision_penalty: Penalty when agents select same non-depot target
            distance_factor: Factor for distance-based rewards
            coverage_bonus: Bonus for covering different areas
            use_difference_reward: Whether to use counterfactual difference rewards
        """
        self.num_agents = num_agents
        self.coordination_bonus = coordination_bonus
        self.collision_penalty = collision_penalty
        self.distance_factor = distance_factor
        self.coverage_bonus = coverage_bonus
        self.use_difference_reward = use_difference_reward
    
    def compute_individual_rewards(
        self,
        actions: List[int],
        env_info: Dict,
        global_reward: float,
        agent_positions: torch.Tensor,
        target_positions: torch.Tensor,
        depot_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute individual rewards for each agent.
        
        Args:
            actions: List of selected target indices per agent
            env_info: Info dict from environment step
            global_reward: Total reward from environment
            agent_positions: (num_agents, 2) agent positions
            target_positions: (num_nodes+1, 2) all target positions (depot first)
            depot_idx: Index of depot (usually 0)
            
        Returns:
            individual_rewards: (num_agents,) reward for each agent
            breakdown: Dict with reward components
        """
        device = agent_positions.device
        rewards = torch.zeros(self.num_agents, device=device)
        breakdown = {
            'base_share': [],
            'distance_cost': [],
            'coordination_bonus': [],
            'collision_penalty': [],
            'coverage_bonus': [],
            'serving_reward': [],
        }
        
        # 1. Base reward share (equal split of global reward)
        base_share = global_reward / self.num_agents
        
        # 2. Distance-based individual cost
        distances = []
        for i, action in enumerate(actions):
            if action < target_positions.size(0):
                target = target_positions[action]
            else:
                target = target_positions[depot_idx]  # fallback to depot
            dist = torch.norm(target - agent_positions[i]).item()
            distances.append(dist)
        
        # 3. Collision detection
        non_depot_actions = [a for a in actions if a != depot_idx]
        has_collision = len(non_depot_actions) != len(set(non_depot_actions))
        
        # 4. Coverage calculation (spatial diversity)
        if len(set(actions)) > 1:
            # Agents targeting different nodes
            unique_targets = list(set(actions))
            if len(unique_targets) >= 2:
                # Calculate spread between targets
                target_coords = [target_positions[a] for a in unique_targets if a < target_positions.size(0)]
                if len(target_coords) >= 2:
                    spread = torch.norm(target_coords[0] - target_coords[1]).item()
                else:
                    spread = 0
            else:
                spread = 0
        else:
            spread = 0
        
        # 5. Compute individual rewards
        served_by_agent = env_info.get('served_by_agent', {})  # agent_idx -> list of served demand ids
        
        for i in range(self.num_agents):
            # Base share
            r_base = base_share
            breakdown['base_share'].append(r_base)
            
            # Distance cost (individual)
            r_dist = -distances[i] * self.distance_factor
            breakdown['distance_cost'].append(r_dist)
            
            # Coordination bonus (if agents select different targets)
            if not has_collision and len(set(actions)) == self.num_agents:
                r_coord = self.coordination_bonus
            else:
                r_coord = 0.0
            breakdown['coordination_bonus'].append(r_coord)
            
            # Collision penalty (if agent's action caused collision)
            if has_collision and actions[i] != depot_idx:
                # Check if this agent's action collides with another
                other_actions = [actions[j] for j in range(self.num_agents) if j != i]
                if actions[i] in other_actions:
                    r_collision = -self.collision_penalty
                else:
                    r_collision = 0.0
            else:
                r_collision = 0.0
            breakdown['collision_penalty'].append(r_collision)
            
            # Coverage bonus (shared among agents with diverse targets)
            if spread > 0:
                r_coverage = self.coverage_bonus * (spread / 20.0)  # normalize by grid size
            else:
                r_coverage = 0.0
            breakdown['coverage_bonus'].append(r_coverage)
            
            # Serving reward (individual credit for demands served)
            served_count = len(served_by_agent.get(i, []))
            r_serve = served_count * 2.0
            breakdown['serving_reward'].append(r_serve)
            
            # Total individual reward
            rewards[i] = r_base + r_dist + r_coord + r_collision + r_coverage + r_serve
        
        return rewards, breakdown
    
    def compute_difference_reward(
        self,
        model: nn.Module,
        obs: Dict,
        mask: torch.Tensor,
        actions: List[int],
        global_reward: float,
        env,
    ) -> torch.Tensor:
        """
        Compute counterfactual difference rewards.
        
        D_i = R(s, a) - R(s, a_{-i}, default_i)
        
        This measures how much agent i contributed compared to taking
        a default action (going to depot).
        
        Args:
            model: The policy model
            obs: Current observation
            mask: Action mask
            actions: Taken actions
            global_reward: Reward with actual actions
            env: Environment for simulation
            
        Returns:
            difference_rewards: (num_agents,) difference reward for each agent
        """
        device = mask.device
        diff_rewards = torch.zeros(self.num_agents, device=device)
        
        # For each agent, compute counterfactual reward with default action
        for i in range(self.num_agents):
            # Counterfactual: agent i goes to depot (default action)
            counterfactual_actions = list(actions)
            counterfactual_actions[i] = 0  # depot
            
            # Estimate counterfactual reward (simplified: just remove serving bonus)
            # In a full implementation, you'd re-simulate the env step
            if actions[i] != 0:
                # Agent actually served something
                counterfactual_reward = global_reward - 1.0  # roughly remove serving bonus
            else:
                counterfactual_reward = global_reward
            
            diff_rewards[i] = global_reward - counterfactual_reward
        
        return diff_rewards


class CoordinationLoss(nn.Module):
    """
    Additional loss term to encourage multi-agent coordination.
    
    Penalizes probability distributions where multiple agents
    have high probability on the same non-depot node.
    """
    
    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        probs: torch.Tensor,
        depot_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute coordination loss.
        
        Args:
            probs: (batch, num_agents, num_nodes+1) probability distributions
            depot_idx: Index of depot in action space
            
        Returns:
            loss: Scalar coordination loss
        """
        batch_size, num_agents, num_actions = probs.shape
        
        if num_agents < 2:
            return torch.tensor(0.0, device=probs.device)
        
        # Exclude depot from collision check
        node_probs = probs[:, :, 1:]  # (batch, agents, nodes)
        
        # Compute pairwise overlap penalty
        # High penalty when two agents both have high prob on same node
        loss = 0.0
        num_pairs = 0
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Element-wise product of probabilities
                # High when both agents want same node
                overlap = node_probs[:, i] * node_probs[:, j]  # (batch, nodes)
                
                # Sum over nodes, mean over batch
                pair_loss = overlap.sum(dim=-1).mean()
                loss += pair_loss
                num_pairs += 1
        
        if num_pairs > 0:
            loss = loss / num_pairs
        
        return self.penalty_weight * loss


class AreaDivisionBonus(nn.Module):
    """
    Bonus for agents dividing the service area effectively.
    
    Encourages agents to specialize in different spatial regions.
    """
    
    def __init__(self, bonus_weight: float = 0.5):
        super().__init__()
        self.bonus_weight = bonus_weight
    
    def forward(
        self,
        probs: torch.Tensor,
        node_positions: torch.Tensor,
        agent_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute area division bonus.
        
        Rewards agents for preferring nodes in their "territory"
        (closer to their current position than other agents).
        
        Args:
            probs: (batch, num_agents, num_nodes+1)
            node_positions: (batch, num_nodes, 2) or (num_nodes, 2)
            agent_positions: (batch, num_agents, 2) or (num_agents, 2)
            
        Returns:
            bonus: Scalar bonus (negative loss)
        """
        batch_size, num_agents, num_actions = probs.shape
        num_nodes = num_actions - 1  # exclude depot
        
        if num_nodes == 0 or num_agents < 2:
            return torch.tensor(0.0, device=probs.device)
        
        # Ensure 3D
        if node_positions.dim() == 2:
            node_positions = node_positions.unsqueeze(0).expand(batch_size, -1, -1)
        if agent_positions.dim() == 2:
            agent_positions = agent_positions.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Node probs (exclude depot)
        node_probs = probs[:, :, 1:]  # (batch, agents, nodes)
        
        # For each node, determine which agent is closest
        # distances: (batch, agents, nodes)
        distances = torch.cdist(
            agent_positions,  # (batch, agents, 2)
            node_positions,   # (batch, nodes, 2)
        )  # (batch, agents, nodes)
        
        # Closest agent per node: (batch, nodes)
        closest_agent = distances.argmin(dim=1)
        
        # Create territory mask: 1 if node is in agent's territory
        territory = torch.zeros_like(node_probs)
        for a in range(num_agents):
            territory[:, a] = (closest_agent == a).float()
        
        # Bonus: agents should have high prob on nodes in their territory
        territory_alignment = (node_probs * territory).sum(dim=-1)  # (batch, agents)
        
        # Mean over agents and batch
        bonus = territory_alignment.mean()
        
        return self.bonus_weight * bonus


def enhanced_reinforce_loss(
    log_probs: List[torch.Tensor],
    probs_list: List[torch.Tensor],
    global_rewards: List[float],
    credit_assigner: CreditAssigner,
    coordination_loss_fn: CoordinationLoss,
    individual_rewards_list: List[torch.Tensor],
    baseline: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Enhanced REINFORCE loss with credit assignment.
    
    Args:
        log_probs: List of log probability tensors per timestep
        probs_list: List of probability tensors per timestep (for coordination loss)
        global_rewards: List of global rewards per timestep
        credit_assigner: CreditAssigner instance
        coordination_loss_fn: CoordinationLoss module
        individual_rewards_list: Pre-computed individual rewards per timestep
        baseline: Optional baseline for variance reduction
        
    Returns:
        loss: Total loss
        info: Dict with loss components
    """
    if not log_probs:
        return torch.tensor(0.0), {}
    
    device = log_probs[0].device
    
    # Compute returns (cumulative future rewards)
    T = len(global_rewards)
    returns = []
    G = 0
    for r in reversed(global_rewards):
        G = r + 0.99 * G  # discount factor
        returns.insert(0, G)
    
    # Baseline
    if baseline is None:
        baseline = sum(returns) / len(returns)
    
    # Policy gradient loss with individual credits
    pg_loss = torch.tensor(0.0, device=device)
    for t in range(T):
        individual_rewards = individual_rewards_list[t]  # (num_agents,)
        log_prob = log_probs[t]  # sum of log probs for this timestep
        
        # Use individual rewards instead of global
        advantage = individual_rewards - baseline / credit_assigner.num_agents
        
        # If log_prob is scalar (summed), distribute by agents
        if log_prob.dim() == 0:
            pg_loss += -log_prob * advantage.mean()
        else:
            pg_loss += -(log_prob * advantage).mean()
    
    pg_loss = pg_loss / T
    
    # Coordination loss
    coord_loss = torch.tensor(0.0, device=device)
    for probs in probs_list:
        coord_loss += coordination_loss_fn(probs)
    coord_loss = coord_loss / len(probs_list) if probs_list else coord_loss
    
    # Total loss
    total_loss = pg_loss + coord_loss
    
    info = {
        'pg_loss': pg_loss.item(),
        'coordination_loss': coord_loss.item(),
        'baseline': baseline,
        'mean_return': sum(returns) / len(returns),
    }
    
    return total_loss, info


# ==============================================================================
# Multi-Vehicle Workload Balance Training
# ==============================================================================

class WorkloadBalanceLoss(nn.Module):
    """
    Loss term to encourage balanced workload distribution among vehicles.
    
    Penalizes scenarios where agents have significantly different:
    - Total travel distances
    - Number of nodes served
    - Total demand served
    
    This encourages the model to learn fair task allocation.
    """
    
    def __init__(
        self,
        distance_weight: float = 1.0,
        node_count_weight: float = 0.5,
        demand_weight: float = 0.5,
    ):
        super().__init__()
        self.distance_weight = distance_weight
        self.node_count_weight = node_count_weight
        self.demand_weight = demand_weight
    
    def forward(
        self,
        agent_distances: torch.Tensor,
        agent_node_counts: torch.Tensor,
        agent_demands: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute workload balance loss.
        
        Args:
            agent_distances: (batch, num_agents) total distance traveled
            agent_node_counts: (batch, num_agents) number of nodes served
            agent_demands: (batch, num_agents) total demand served
            
        Returns:
            loss: Scalar balance loss
            metrics: Dict with balance metrics
        """
        # Normalize metrics to same scale
        dist_mean = agent_distances.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        dist_std = agent_distances.std(dim=-1)
        dist_cv = dist_std / dist_mean.squeeze(-1)  # Coefficient of variation
        
        count_mean = agent_node_counts.float().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        count_std = agent_node_counts.float().std(dim=-1)
        count_cv = count_std / count_mean.squeeze(-1)
        
        demand_mean = agent_demands.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        demand_std = agent_demands.std(dim=-1)
        demand_cv = demand_std / demand_mean.squeeze(-1)
        
        # Combined loss: penalize high variance (CV)
        loss = (
            self.distance_weight * dist_cv.mean() +
            self.node_count_weight * count_cv.mean() +
            self.demand_weight * demand_cv.mean()
        )
        
        metrics = {
            'distance_cv': dist_cv.mean().item(),
            'node_count_cv': count_cv.mean().item(),
            'demand_cv': demand_cv.mean().item(),
            'balance_loss': loss.item(),
        }
        
        return loss, metrics


class WorkloadTracker:
    """
    Tracks per-agent workload statistics during an episode.
    
    Used to compute workload balance metrics and rewards.
    """
    
    def __init__(self, num_agents: int, device: torch.device):
        self.num_agents = num_agents
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset trackers for new episode."""
        self.total_distances = torch.zeros(self.num_agents, device=self.device)
        self.node_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
        self.demand_served = torch.zeros(self.num_agents, device=self.device)
        self.step_count = 0
    
    def update(
        self,
        actions: List[int],
        distances: List[float],
        demands_served: Optional[Dict[int, float]] = None,
    ):
        """
        Update workload statistics after a step.
        
        Args:
            actions: List of action indices per agent (0=depot)
            distances: List of distances traveled per agent
            demands_served: Dict mapping agent_idx -> demand served this step
        """
        self.step_count += 1
        
        for i, (action, dist) in enumerate(zip(actions, distances)):
            self.total_distances[i] += dist
            if action != 0:  # Not depot
                self.node_counts[i] += 1
            
            if demands_served and i in demands_served:
                self.demand_served[i] += demands_served[i]
    
    def get_balance_metrics(self) -> Dict:
        """
        Compute balance metrics for current episode.
        
        Returns:
            Dict with balance statistics
        """
        # Coefficient of variation (lower = more balanced)
        dist_mean = self.total_distances.mean().item()
        dist_std = self.total_distances.std().item()
        dist_cv = dist_std / max(dist_mean, 1e-6)
        
        count_mean = self.node_counts.float().mean().item()
        count_std = self.node_counts.float().std().item()
        count_cv = count_std / max(count_mean, 1e-6)
        
        demand_mean = self.demand_served.mean().item()
        demand_std = self.demand_served.std().item()
        demand_cv = demand_std / max(demand_mean, 1e-6)
        
        # Overall balance score (0-1, higher = more balanced)
        balance_score = 1.0 - min((dist_cv + count_cv + demand_cv) / 3, 1.0)
        
        return {
            'total_distances': self.total_distances.tolist(),
            'node_counts': self.node_counts.tolist(),
            'demand_served': self.demand_served.tolist(),
            'distance_cv': dist_cv,
            'node_count_cv': count_cv,
            'demand_cv': demand_cv,
            'balance_score': balance_score,
        }
    
    def get_balance_reward(self, scale: float = 2.0) -> float:
        """
        Compute balance reward for episode end.
        
        Args:
            scale: Scaling factor for reward
            
        Returns:
            Balance reward (positive for good balance)
        """
        metrics = self.get_balance_metrics()
        return scale * metrics['balance_score']
    
    def get_individual_balance_rewards(self, scale: float = 1.0) -> torch.Tensor:
        """
        Compute per-agent balance rewards.
        
        Agents with workload closer to mean get higher reward.
        
        Args:
            scale: Scaling factor
            
        Returns:
            (num_agents,) reward tensor
        """
        # Distance from mean workload (normalized)
        dist_mean = self.total_distances.mean()
        dist_deviation = torch.abs(self.total_distances - dist_mean)
        dist_reward = 1.0 - dist_deviation / dist_mean.clamp(min=1e-6)
        
        count_mean = self.node_counts.float().mean()
        count_deviation = torch.abs(self.node_counts.float() - count_mean)
        count_reward = 1.0 - count_deviation / count_mean.clamp(min=1e-6)
        
        # Combined reward (clamp to reasonable range)
        individual_rewards = (dist_reward + count_reward) / 2.0
        individual_rewards = individual_rewards.clamp(min=-1.0, max=1.0) * scale
        
        return individual_rewards


def compute_balance_bonus(
    agent_distances: List[float],
    agent_node_counts: List[int],
    target_ratio: float = 1.0,
) -> float:
    """
    Compute balance bonus for a set of agent statistics.
    
    Args:
        agent_distances: Total distance per agent
        agent_node_counts: Number of nodes served per agent
        target_ratio: Target ratio between agents (1.0 = equal)
        
    Returns:
        Bonus value (positive for balanced, negative for unbalanced)
    """
    if not agent_distances or not agent_node_counts:
        return 0.0
    
    n_agents = len(agent_distances)
    if n_agents < 2:
        return 0.0
    
    # Compute imbalance ratio
    dist_max = max(agent_distances) if max(agent_distances) > 0 else 1.0
    dist_min = min(agent_distances) if min(agent_distances) > 0 else 0.0
    dist_ratio = dist_min / dist_max if dist_max > 0 else 0.0
    
    count_max = max(agent_node_counts) if max(agent_node_counts) > 0 else 1
    count_min = min(agent_node_counts) if min(agent_node_counts) > 0 else 0
    count_ratio = count_min / count_max if count_max > 0 else 0.0
    
    # Balance score: 1.0 when perfectly balanced, 0.0 when completely unbalanced
    balance_score = (dist_ratio + count_ratio) / 2.0
    
    # Return bonus (positive when balanced)
    return balance_score * 2.0 - 1.0  # Range: [-1, 1]


# Convenience function for training loop
def compute_step_credits(
    credit_assigner: CreditAssigner,
    actions: List[int],
    env_info: Dict,
    global_reward: float,
    agent_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute individual credits for a single step.
    
    Wrapper for use in training loops.
    """
    rewards, _ = credit_assigner.compute_individual_rewards(
        actions=actions,
        env_info=env_info,
        global_reward=global_reward,
        agent_positions=agent_positions,
        target_positions=target_positions,
    )
    return rewards


def compute_step_credits_with_balance(
    credit_assigner: CreditAssigner,
    workload_tracker: WorkloadTracker,
    actions: List[int],
    env_info: Dict,
    global_reward: float,
    agent_positions: torch.Tensor,
    target_positions: torch.Tensor,
    balance_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute individual credits with balance consideration.
    
    This version adds balance rewards to encourage fair workload distribution.
    
    Args:
        credit_assigner: CreditAssigner instance
        workload_tracker: WorkloadTracker to get balance stats
        actions: Action indices per agent
        env_info: Environment info dict
        global_reward: Global step reward
        agent_positions: Agent positions
        target_positions: Target positions
        balance_weight: Weight for balance component
        
    Returns:
        individual_rewards: (num_agents,) tensor
    """
    # Base individual rewards
    base_rewards, _ = credit_assigner.compute_individual_rewards(
        actions=actions,
        env_info=env_info,
        global_reward=global_reward,
        agent_positions=agent_positions,
        target_positions=target_positions,
    )
    
    # Add balance rewards
    balance_rewards = workload_tracker.get_individual_balance_rewards(scale=balance_weight)
    
    return base_rewards + balance_rewards
