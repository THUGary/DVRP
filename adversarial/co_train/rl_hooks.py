from __future__ import annotations
"""Reusable planner training hooks for co-evolution.

This module provides a simple REINFORCE-style hook that you can pass into
coevolution_loop(planner_update_hook=...) to train the model-based planner
with policy gradient on on-policy rollouts sampled from the environment.

The hook assumes a ModelPlanner with a DVRPNet at `planner._model` and uses
utility functions from models.planner_model.model and the RL helpers in
training/planner/train_rl_planner.py for sampling decisions with log-probs.
"""
from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F


def _towards_step(cur: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[int, int]:
    cx, cy = cur; dx = 0; dy = 0
    if dst[0] != cx:
        dx = 1 if dst[0] > cx else -1
    elif dst[1] != cy:
        dy = 1 if dst[1] > cy else -1
    return dx, dy


def reinforce_planner_hook(planner, ctx: Dict[str, Any]) -> None:
    """A minimal REINFORCE update for the model-based planner.

    Contract:
    - planner: expected to be a ModelPlanner with fields `_model`, `d_model`,
      `lateness_lambda`, and `full_capacity`.
    - ctx: should contain keys
        env: GridEnvironment
        rng: random.Random
        opt_planner: Optional[torch.optim.Optimizer]
        diffusion_model, condition, base_cfg, device: for (re)generating demands
    Behavior:
    - Generates one episode with demands sampled from the current diffusion model
      (already loaded by the main loop for the selected generator version).
    - At each env step, samples actions for all agents using the planner model,
      collects the joint log-prob, accumulates rewards, and finally applies a
      REINFORCE loss with discounted returns.
    """
    env = ctx["env"]
    device: torch.device = ctx.get("device", torch.device("cpu"))  # type: ignore[assignment]
    opt = ctx.get("opt_planner")

    # Lazily create an optimizer if not provided
    if opt is None and hasattr(planner, "_model") and hasattr(planner._model, "parameters"):
        opt = torch.optim.AdamW(planner._model.parameters(), lr=1e-4, weight_decay=1e-6)
        ctx["opt_planner"] = opt

    # 1) Generate demands using the (already loaded) diffusion model
    diff_model = ctx.get("diffusion_model")
    condition = ctx.get("condition")
    base_cfg = ctx.get("base_cfg")
    if diff_model is None or condition is None or base_cfg is None:
        # If context is incomplete, skip update silently
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
    except Exception:
        return

    # 2) Reset env and inject demands
    obs = env.reset()
    if hasattr(env, "_state") and getattr(env, "_state") is not None:
        try:
            from agent.generator.base import Demand as _Demand
            env._state.demands.extend([_Demand(x=d[0], y=d[1], t=d[2], c=d[3], end_t=d[4]) for d in demands_list])
        except Exception:
            pass
    if hasattr(env, "_obs"):
        obs = env._obs()

    # 3) Rollout one episode while collecting log-probs and rewards
    logp_steps: List[torch.Tensor] = []
    rewards: List[float] = []

    # Imports for features and action sampling from the planner model
    from models.planner_model.model import prepare_features
    from training.planner.train_rl_planner import select_targets_with_sampling

    lateness_lambda = getattr(planner, "lateness_lambda", 0.0)
    full_capacity = getattr(planner, "full_capacity", None)
    if full_capacity is None:
        # ModelPlanner requires full_capacity to compute cap_full
        return

    done = False
    while not done:
        # Build features for DVRPNet
        nodes_list = obs["demands"]
        t_now = obs["time"]
        depot_xy = tuple(obs["depot"])  # (x,y)
        mask = [False] * len(nodes_list)
        feats = prepare_features(
            nodes=[nodes_list],  # [1,N,5]
            node_mask=[mask],
            depot=[(depot_xy[0], depot_xy[1], t_now)],  # [1,1,3]
            d_model=getattr(planner._model, "d_model", 128),
            device=device,
        )
        # Agents tensor [1,A,4] with time
        import torch as _torch
        agents = obs["agent_states"]
        agents_tensor = _torch.tensor([[ (a[0], a[1], a[2], t_now) for a in agents ]], dtype=_torch.float32, device=device)

        # cap_full: [1,A]
        cap_full = _torch.full((1, agents_tensor.size(1)), float(full_capacity), dtype=_torch.float32, device=device)

        # Sample one target per agent with log-probs
        sel, dest_xy, log_probs, _value = select_targets_with_sampling(
            planner._model,  # DVRPNet
            feats,
            agents_tensor,
            lateness_lambda,
            cap_full,
        )
        # Sum log-probs across agents for this step
        step_logp = log_probs.sum()

        # Convert destination coords into one-grid actions
        actions = []
        for idx, (ax, ay, _s) in enumerate(agents):
            dst = (int(dest_xy[0, idx, 0].item()), int(dest_xy[0, idx, 1].item()))
            actions.append(_towards_step((ax, ay), dst))

        # Env step
        obs, reward, done, _info = env.step(actions)
        rewards.append(float(reward))
        logp_steps.append(step_logp)

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
    import math
    mean_r = float(sum(returns) / max(1, len(returns)))
    var_r = float(sum((x - mean_r) * (x - mean_r) for x in returns) / max(1, len(returns)))
    std_r = math.sqrt(var_r + 1e-8)
    returns_t = torch.tensor([(r - mean_r) / (std_r if std_r > 0 else 1.0) for r in returns], dtype=torch.float32, device=device)

    # 5) REINFORCE loss: -sum_t logp_t * G_t
    logp_t = torch.stack(logp_steps).to(device)
    loss = -(logp_t * returns_t).sum() / returns_t.numel()

    if opt is None:
        return
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(planner._model.parameters(), 1.0)
    opt.step()
