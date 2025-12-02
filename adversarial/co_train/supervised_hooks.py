from typing import Any, Dict, Tuple
import torch

def supervised_planner_hook(planner: Any, ctx: Dict[str, Any]) -> None:
    """Default small supervised planner update extracted from train_coevolution.

    Supports V2Planner (POMO-based architecture).

    Expects ctx to contain:
      - env: environment
      - base_cfg: base configuration (capacity, generator params, etc.)
      - diffusion_model: diffusion model used to sample demands
      - condition: diffusion condition object
      - device: torch.device to run on
      - opt_planner: optional optimizer for planner._model

    This mirrors the previous inlined behaviour: sample a small set of demands,
    ask the rule-based teacher for a one-step plan (k=1) and perform a single
    cross-entropy gradient step on the planner network.
    """
    from agent.planner.v2_planner import V2Planner
    
    if isinstance(planner, V2Planner):
        supervised_v2_planner_hook(planner, ctx)
    else:
        print(f"[Supervised Hook] Unsupported planner type: {type(planner).__name__}")


def supervised_v2_planner_hook(planner: Any, ctx: Dict[str, Any]) -> None:
    """Supervised training hook for V2Planner (dynamic mode).
    
    Uses rule-based teacher to provide target actions, then trains
    the dynamic adapter with cross-entropy loss.
    """
    from agent.planner.v2_planner import V2Planner
    
    if not isinstance(planner, V2Planner):
        return
    
    if planner.mode != "dynamic":
        # Static mode doesn't need supervised training with teacher
        print("[Supervised Hook] V2Planner in static mode, skipping supervised update")
        return
    
    env = ctx["env"]
    base_cfg = ctx["base_cfg"]
    diffusion_model = ctx["diffusion_model"]
    condition = ctx.get("condition")
    device = ctx.get("device", torch.device("cpu"))
    opt_planner = ctx.get("opt_planner")

    from agent.planner.rule_planner import RuleBasedPlanner
    from training.generator.adversarial_trainer import _generate_demands as _gen

    teacher = RuleBasedPlanner(full_capacity=base_cfg.capacity)
    obs = env.reset()

    # Generate demands using the diffusion model
    demands_list = _gen(diffusion_model, condition, {
        'width': base_cfg.width,
        'height': base_cfg.height,
        'max_time': base_cfg.max_time,
        'max_c': base_cfg.generator_params['max_c'],
        'min_lifetime': base_cfg.generator_params['min_lifetime'],
        'max_lifetime': base_cfg.generator_params['max_lifetime'],
        'total_demand': base_cfg.generator_params['total_demand']
    })

    # Inject demands into env
    if hasattr(env, "_state") and env._state is not None:
        from agent.generator.base import Demand as _Demand
        def _to_demand(raw: Tuple[int, ...]) -> _Demand:
            service_time = int(raw[5]) if len(raw) > 5 else 0
            return _Demand(x=raw[0], y=raw[1], t=raw[2], c=raw[3], end_t=raw[4], service_time=service_time)
        env._state.demands.extend([_to_demand(d) for d in demands_list])
    obs = env._obs() if hasattr(env, "_obs") else obs

    # Get teacher targets
    agent_states_raw = obs["agent_states"]
    agents_state_objs = [
        type("AS", (), {"x": a[0], "y": a[1], "s": a[2]}) for a in agent_states_raw
    ]
    targets = teacher.plan(
        observations=obs["demands"],
        agent_states=agents_state_objs,
        depot=tuple(obs["depot"]),
        t=obs["time"],
        horizon=1,
    )

    nodes_list = obs["demands"]
    if not nodes_list:
        return
    
    N = len(nodes_list)
    t_now = obs["time"]
    depot = tuple(obs["depot"])
    num_agents = env.num_agents
    
    # Ensure model is loaded
    planner._ensure_model_loaded()
    model = planner._model
    
    # Normalization constants
    coord_norm = planner.coord_norm
    capacity_norm = planner.capacity_norm
    time_norm = planner.time_norm
    
    # Prepare tensors for model
    depot_xy = torch.tensor([[[depot[0] / coord_norm, depot[1] / coord_norm]]], 
                            dtype=torch.float32, device=device)
    
    node_coords = [[n[0] / coord_norm, n[1] / coord_norm] for n in nodes_list]
    node_demands = [n[3] / capacity_norm for n in nodes_list]
    node_xy = torch.tensor([node_coords], dtype=torch.float32, device=device)
    node_demand = torch.tensor([node_demands], dtype=torch.float32, device=device)
    
    # Optional: deadlines
    node_deadline = torch.tensor([[n[4] for n in nodes_list]], dtype=torch.float32, device=device)
    time_now_tensor = torch.tensor([t_now], dtype=torch.float32, device=device)
    
    # Agent states: (1, A, 4)
    agent_data = []
    for ax, ay, as_ in agent_states_raw:
        agent_data.append([
            ax / coord_norm,
            ay / coord_norm,
            as_ / capacity_norm,
            t_now / time_norm,
        ])
    agent_states_tensor = torch.tensor([agent_data], dtype=torch.float32, device=device)
    
    # Mask: (1, A, N+1)
    ninf_mask = torch.zeros(1, num_agents, N + 1, device=device)
    
    # Enable training
    model.train()
    
    # Forward pass with full probabilities for supervised learning
    selected, probs = model.forward_with_full_probs(
        depot_xy, node_xy, node_demand,
        agent_states_tensor, ninf_mask,
        node_deadline=node_deadline,
        time_now=time_now_tensor,
    )
    
    # Build labels from teacher targets
    # Map (x, y) -> node index (0=depot, 1..N=nodes)
    xy_to_idx = {
        (int(node_xy[0, i, 0].item() * coord_norm), int(node_xy[0, i, 1].item() * coord_norm)): i + 1 
        for i in range(N)
    }
    # Also add depot
    xy_to_idx[(depot[0], depot[1])] = 0
    
    labels = torch.zeros(1, num_agents, dtype=torch.long, device=device)
    for a_idx, q in enumerate(targets):
        if len(q) == 0:
            labels[0, a_idx] = 0  # depot
        else:
            tgt_xy = q[0]
            labels[0, a_idx] = xy_to_idx.get((int(tgt_xy[0]), int(tgt_xy[1])), 0)
    
    # Cross-entropy loss
    # probs: (1, A, N+1) -> (A, N+1)
    logits = torch.log(probs + 1e-8).squeeze(0)  # (A, N+1)
    loss = torch.nn.functional.cross_entropy(logits, labels.squeeze(0))
    
    if opt_planner is not None:
        opt_planner.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
        opt_planner.step()
    
    model.eval()
    
    # Advance env one step
    actions = []
    for i, (x, y, s) in enumerate(agent_states_raw):
        if len(targets[i]) == 0:
            actions.append((0, 0))
        else:
            tx, ty = targets[i][0]
            dx = 1 if tx > x else (-1 if tx < x else 0)
            dy = 1 if ty > y else (-1 if ty < y else 0)
            actions.append((dx, dy))
    env.step(actions)
