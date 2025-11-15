from typing import Any, Dict
import torch

def supervised_planner_hook(planner: Any, ctx: Dict[str, Any]) -> None:
    """Default small supervised planner update extracted from train_coevolution.

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
    env = ctx["env"]
    base_cfg = ctx["base_cfg"]
    diffusion_model = ctx["diffusion_model"]
    condition = ctx.get("condition")
    device = ctx.get("device", torch.device("cpu"))
    opt_planner = ctx.get("opt_planner")

    # Local imports (keep lightweight at module import time)
    from agent.planner.rule_planner import RuleBasedPlanner
    from agent.controller import RuleBasedController
    from models.planner_model.model import prepare_features
    from training.generator.adversarial_trainer import _generate_demands as _gen

    teacher = RuleBasedPlanner(full_capacity=base_cfg.capacity)
    controller = RuleBasedController(**base_cfg.controller_params)
    obs = env.reset()

    # generate demands using the diffusion model helper (same args as prior)
    demands_list = _gen(diffusion_model, condition, {
        'width': base_cfg.width,
        'height': base_cfg.height,
        'max_time': base_cfg.max_time,
        'max_c': base_cfg.generator_params['max_c'],
        'min_lifetime': base_cfg.generator_params['min_lifetime'],
        'max_lifetime': base_cfg.generator_params['max_lifetime'],
        'total_demand': base_cfg.generator_params['total_demand']
    })

    # If env keeps a private state with demands, append sampled demands there
    if hasattr(env, "_state") and env._state is not None:
        from agent.generator.base import Demand as _Demand
        env._state.demands.extend([_Demand(x=d[0], y=d[1], t=d[2], c=d[3], end_t=d[4]) for d in demands_list])
    obs = env._obs() if hasattr(env, "_obs") else obs

    # build one supervised sample at time t with k=1 target from teacher
    agent_states = obs["agent_states"]
    agents_state_objs = [
        type("AS", (), {"x": a[0], "y": a[1], "s": a[2]}) for a in agent_states
    ]
    targets = teacher.plan(
        observations=obs["demands"],
        agent_states=agents_state_objs,  # type: ignore[arg-type]
        depot=tuple(obs["depot"]),
        t=obs["time"],
        horizon=1,
    )

    # Prepare features and labels
    feats = prepare_features(nodes=[obs["demands"]], node_mask=[[False]*len(obs["demands"])], depot=[(obs["depot"][0], obs["depot"][1], obs["time"])], d_model=getattr(planner._model, "d_model", 128), device=device)  # type: ignore[attr-defined]
    _torch = torch
    agents_tensor = _torch.tensor([[ (a[0], a[1], a[2], obs["time"]) for a in agent_states ]], dtype=_torch.float32, device=device)
    enc_nodes, enc_depot, node_mask = planner._model.encoder(feats)  # type: ignore[attr-defined]
    enc_agents = planner._model.encoder.encode_agents(agents_tensor)  # type: ignore[attr-defined]
    logits = planner._model.decode(enc_nodes=enc_nodes, enc_depot=enc_depot, node_mask=node_mask, enc_agents=enc_agents, agents_tensor=agents_tensor, nodes=feats.get("nodes"), lateness_lambda=getattr(planner, "lateness_lambda", 0.0), history_indices=None)  # type: ignore[attr-defined]

    # labels from teacher targets: map coord to index (0=depot, 1..N nodes)
    nodes = feats["nodes"]
    B, N, _ = nodes.shape
    A = agents_tensor.size(1)
    labels = _torch.zeros((1, A), dtype=_torch.long, device=device)
    # create mapping
    xy_to_idx = { (int(nodes[0, i, 0].item()), int(nodes[0, i, 1].item())): i+1 for i in range(N) }
    for a_idx, q in enumerate(targets):
        if len(q) == 0:
            labels[0, a_idx] = 0
        else:
            tgt_xy = q[0]
            labels[0, a_idx] = xy_to_idx.get((int(tgt_xy[0]), int(tgt_xy[1])), 0)
    loss = _torch.nn.functional.cross_entropy(logits.view(1*A, -1), labels.view(-1))
    if opt_planner is not None:
        opt_planner.zero_grad(); loss.backward(); _torch.nn.utils.clip_grad_norm_(planner._model.parameters(), 1.0); opt_planner.step()  # type: ignore[attr-defined]

    # advance env one step to avoid infinite loops in some envs
    actions = []
    for i, (x, y, s) in enumerate(agent_states):
        if len(targets[i]) == 0:
            actions.append((0,0))
        else:
            tx, ty = targets[i][0]
            dx = 1 if tx> x else (-1 if tx < x else 0)
            dy = 1 if ty> y else (-1 if ty < y else 0)
            actions.append((dx, dy))
    env.step(actions)
