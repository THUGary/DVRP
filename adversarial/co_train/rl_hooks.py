from __future__ import annotations
"""Reusable planner training hooks for co-evolution.

This module provides REINFORCE-style hooks that you can pass into
coevolution_loop(planner_update_hook=...) to train the planner model
with policy gradient on on-policy rollouts sampled from the environment.

Supports V2Planner (POMO-based architecture).
"""
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from collections import deque
import math


def _towards_step(cur: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[int, int]:
    """Compute single step direction from cur to dst."""
    cx, cy = cur
    dx, dy = 0, 0
    if dst[0] != cx:
        dx = 1 if dst[0] > cx else -1
    elif dst[1] != cy:
        dy = 1 if dst[1] > cy else -1
    return dx, dy


def reinforce_v2_planner_hook(planner, ctx: Dict[str, Any]) -> None:
    """REINFORCE update for V2Planner (dynamic mode with adapter).

    This hook trains the adapter parameters of DynamicVRPModel using
    policy gradient (REINFORCE) with baseline.

    Contract:
    - planner: V2Planner with mode="dynamic" (has _model as DynamicVRPModel)
    - ctx: should contain keys
        env: GridEnvironment
        rng: random.Random
        opt_planner: Optional[torch.optim.Optimizer]
        diffusion_model, condition, base_cfg, device: for demand generation
    """
    from agent.planner.v2_planner import V2Planner
    
    # Validate planner type
    if not isinstance(planner, V2Planner):
        print("[RL Hook] Warning: Expected V2Planner, got", type(planner).__name__)
        return
    
    if planner.mode != "dynamic":
        print("[RL Hook] Warning: V2Planner in static mode, skipping RL update")
        return
    
    env = ctx["env"]
    device: torch.device = ctx.get("device", torch.device("cpu"))
    opt = ctx.get("opt_planner")
    
    # Ensure model is loaded
    planner._ensure_model_loaded()
    model = planner._model
    
    # Lazily create optimizer if not provided
    if opt is None:
        opt = torch.optim.AdamW(model.get_trainable_params(), lr=1e-4, weight_decay=1e-6)
        ctx["opt_planner"] = opt

    # 1) Generate demands using the diffusion model
    diff_model = ctx.get("diffusion_model")
    condition = ctx.get("condition")
    base_cfg = ctx.get("base_cfg")
    
    if diff_model is None or condition is None or base_cfg is None:
        return

    try:
        from training.generator.adversarial_trainer import _generate_demands as _gen
        demands_list = _gen(diff_model, condition, {
            'width': base_cfg.width,
            'height': base_cfg.height,
            'max_time': base_cfg.max_time,
            'max_c': base_cfg.generator_params['max_c'],
            'min_lifetime': base_cfg.generator_params['min_lifetime'],
            'max_lifetime': base_cfg.generator_params['max_lifetime'],
            'total_demand': base_cfg.generator_params['total_demand']
        })
    except Exception as e:
        print(f"[RL Hook] Demand generation failed: {e}")
        return

    if not demands_list:
        return

    # 2) Reset env and inject demands
    obs = env.reset()
    if hasattr(env, "_state") and getattr(env, "_state") is not None:
        try:
            from agent.generator.base import Demand as _Demand
            def _to_demand(raw: Tuple[int, ...]) -> _Demand:
                service_time = int(raw[5]) if len(raw) > 5 else 0
                return _Demand(x=raw[0], y=raw[1], t=raw[2], c=raw[3], end_t=raw[4], service_time=service_time)
            env._state.demands.extend([_to_demand(d) for d in demands_list])
        except Exception:
            pass
    if hasattr(env, "_obs"):
        obs = env._obs()

    # 3) Rollout one episode collecting log-probs and rewards
    logp_steps: List[torch.Tensor] = []
    rewards: List[float] = []
    
    # Enable training mode for adapter
    model.train()
    
    # Normalization constants
    coord_norm = planner.coord_norm
    capacity_norm = planner.capacity_norm
    time_norm = planner.time_norm
    num_agents = env.num_agents
    
    done = False
    while not done:
        nodes_list = obs["demands"]
        if nodes_list is None or len(nodes_list) == 0:
            break
            
        t_now = obs["time"]
        depot = tuple(obs["depot"])
        agents = obs["agent_states"]
        N = len(nodes_list)
        
        # Prepare tensors for model
        # Depot: (1, 1, 2)
        depot_xy = torch.tensor([[[depot[0] / coord_norm, depot[1] / coord_norm]]], 
                                dtype=torch.float32, device=device)
        
        # Nodes: (1, N, 2) and demands: (1, N)
        node_coords = [[n[0] / coord_norm, n[1] / coord_norm] for n in nodes_list]
        node_demands = [n[3] / capacity_norm for n in nodes_list]
        node_xy = torch.tensor([node_coords], dtype=torch.float32, device=device)
        node_demand = torch.tensor([node_demands], dtype=torch.float32, device=device)
        
        # Optional: deadlines
        node_deadline = torch.tensor([[n[4] for n in nodes_list]], dtype=torch.float32, device=device)
        time_now_tensor = torch.tensor([t_now], dtype=torch.float32, device=device)
        
        # Agent states: (1, A, 4)
        agent_data = []
        for ax, ay, as_ in agents:
            agent_data.append([
                ax / coord_norm,
                ay / coord_norm,
                as_ / capacity_norm,
                t_now / time_norm,
            ])
        agent_states_tensor = torch.tensor([agent_data], dtype=torch.float32, device=device)
        
        # Mask: (1, A, N+1)
        ninf_mask = torch.zeros(1, num_agents, N + 1, device=device)
        
        # Forward pass - get selected and probs
        # Note: DynamicVRPModel.forward returns (selected, prob_of_selected)
        # where prob_of_selected is (batch, n_agents) - the probability of selected action
        selected, prob_selected = model(
            depot_xy, node_xy, node_demand,
            agent_states_tensor, ninf_mask,
            node_deadline=node_deadline,
            time_now=time_now_tensor,
        )
        
        # Compute log probabilities for selected actions
        # prob_selected: (1, A) - probability of the selected action for each agent
        log_probs = torch.log(prob_selected + 1e-8)
        step_logp = log_probs.sum()
        logp_steps.append(step_logp)
        
        # Convert selected indices to actions
        actions = []
        for a_idx in range(num_agents):
            idx = selected[0, a_idx].item()
            if idx == 0:
                # Return to depot
                target = depot
            else:
                node_idx = idx - 1
                if node_idx < N:
                    target = (nodes_list[node_idx][0], nodes_list[node_idx][1])
                else:
                    target = depot
            
            ax, ay, _ = agents[a_idx]
            dx, dy = _towards_step((ax, ay), target)
            actions.append((dx, dy))
        
        # Environment step
        obs, reward, done, _ = env.step(actions)
        rewards.append(float(reward))

    model.eval()
    
    if not logp_steps:
        return

    # 4) Compute discounted returns
    gamma = 0.99
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()

    # Normalize returns for stability
    mean_r = sum(returns) / max(1, len(returns))
    var_r = sum((x - mean_r) ** 2 for x in returns) / max(1, len(returns))
    std_r = math.sqrt(var_r + 1e-8)
    returns_t = torch.tensor(
        [(r - mean_r) / max(std_r, 1e-8) for r in returns], 
        dtype=torch.float32, device=device
    )

    # 5) REINFORCE loss
    logp_t = torch.stack(logp_steps).to(device)
    loss = -(logp_t * returns_t).sum() / returns_t.numel()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
    opt.step()


def reinforce_planner_hook(planner, ctx: Dict[str, Any]) -> None:
    """Universal REINFORCE hook for V2Planner.
    
    This is the main entry point for REINFORCE training of the planner.
    """
    from agent.planner.v2_planner import V2Planner
    
    if isinstance(planner, V2Planner):
        reinforce_v2_planner_hook(planner, ctx)
    else:
        print(f"[RL Hook] Unsupported planner type: {type(planner).__name__}")
