from __future__ import annotations
"""Main co-evolution / co-training loop.

This script-like module exposes `coevolution_loop` which alternates training
between the planner model (DVRPNet) and the diffusion demand generator.

High-level algorithm (simplified pseudo-fictitious play):
  for cycle in range(num_cycles):
      # 1. Planner phase: train/update planner against mixture of generator versions
      for planner_epoch in range(P_planner_epochs):
           sample generator version v ~ Scheduler(policy)
           generate demands using v
           collect planner training batch (supervised or RL) => update planner
      save planner checkpoint

      # 2. Generator phase: adversarial update against current planner
      for gen_epoch in range(P_generator_epochs):
           sample latent noise => demands
           rollout env with planner => reward
           diffusion loss * advantage => update generator
      save generator checkpoint, add to registry

This scaffold calls existing builders and leaves concrete batch construction /
loss computations to hooks you can pass in. Default hooks are NO-OP so you can
gradually fill them in.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import torch, random, os

from adversarial.builders import build_env, build_planner, build_diffusion
from training.generator.adversarial_trainer import DiffusionAdversarialTrainer, AdvConfig
from .version_registry import GeneratorVersionRegistry
from .match_scheduler import PlannerTrainingScheduler


@dataclass
class CoevolutionConfig:
    num_cycles: int = 5
    planner_epochs_per_cycle: int = 10
    generator_epochs_per_cycle: int = 10
    scheduler_policy: str = "latest_biased"  # or "uniform"
    latest_bias: float = 0.7
    device: str = "cuda"
    seed: int = 42
    save_dir: str = "checkpoints/coevolution"
    # 在 planner phase 采样 generator 版本时，优先选择最新版本的概率
    # (余下概率从历史版本中随机选择)。取值范围 [0,1]
    sample_latest_prob: float = 0.7


def coevolution_loop(
    cfg: CoevolutionConfig,
    planner_type: str = "model",
    planner_ckpt_init: Optional[str] = None,
    diffusion_ckpt_init: Optional[str] = None,
    planner_update_hook: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    generator_metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Run co-training cycles.

    planner_update_hook(planner, context) is invoked per planner epoch with:
      context = {"env": env, "generator_version": gv, "rng": rng}
    You can implement supervised dataset augmentation or an RL step there.

    generator_metrics_hook(metrics) invoked when a new generator version is added.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    rng = random.Random(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Build env & initial planner/generator
    env, base_cfg = build_env()
    planner = build_planner(planner_type, base_cfg, device, planner_ckpt_init)
    diffusion_model, condition = build_diffusion(base_cfg, device, diffusion_ckpt_init)
    adv_cfg = AdvConfig(randomize_depot=True, lr=1e-4, normalize_reward=False)
    gen_trainer = DiffusionAdversarialTrainer(env, diffusion_model, condition, base_cfg, device, adv_cfg)

    registry = GeneratorVersionRegistry()
    # Register initial generator snapshot
    init_gen_path = os.path.join(cfg.save_dir, "generator_init.pth")
    torch.save(diffusion_model.state_dict(), init_gen_path)
    registry.add(init_gen_path, metrics={"cycle": 0, "tag": "init"})

    scheduler = PlannerTrainingScheduler(registry, policy=cfg.scheduler_policy, latest_bias=cfg.latest_bias)

    # Lazy imports for teacher and feature utils
    from agent.planner.rule_planner import RuleBasedPlanner
    from agent.controller import RuleBasedController
    from models.planner_model.model import prepare_features

    opt_planner = None
    if hasattr(planner, "_model") and hasattr(planner._model, "parameters"):
        opt_planner = torch.optim.AdamW(planner._model.parameters(), lr=1e-3, weight_decay=1e-6)

    for cycle in range(1, cfg.num_cycles + 1):
        print(f"=== Cycle {cycle}/{cfg.num_cycles} ===")
        # ---- Planner phase ----
        for pe in range(1, cfg.planner_epochs_per_cycle + 1):
            # Choose a generator version to generate training data.
            # With probability cfg.sample_latest_prob pick the latest version,
            # otherwise pick a random historical version (exclude latest).
            versions = registry.list()
            if not versions:
                raise RuntimeError("No generator versions in registry")
            if len(versions) == 1:
                gv = versions[0]
            else:
                if rng.random() < cfg.sample_latest_prob:
                    gv = versions[-1]
                else:
                    gv = rng.choice(versions[:-1])
            # Load generator weights into diffusion_model for demand sampling
            state = gv.load(device)
            try:
                diffusion_model.load_state_dict(state, strict=False)
            except Exception:
                # Accept partial load (e.g., state might be wrapped)
                if isinstance(state, dict) and "model" in state:
                    diffusion_model.load_state_dict(state["model"], strict=False)
            diffusion_model.eval()
            # Hook for planner update (user implements training step)
            if planner_update_hook:
                planner_update_hook(planner, {"env": env, "generator_version": gv, "rng": rng})
            else:
                # Default: one mini supervised step using a greedy teacher and k=1 labels
                teacher = RuleBasedPlanner(full_capacity=base_cfg.capacity)
                controller = RuleBasedController(**base_cfg.controller_params)
                obs = env.reset()
                # inject demands via trainer helper then sync env state
                from training.generator.adversarial_trainer import _generate_demands as _gen
                demands_list = _gen(diffusion_model, condition, {
                    'width': base_cfg.width,
                    'height': base_cfg.height,
                    'max_time': base_cfg.max_time,
                    'max_c': base_cfg.generator_params['max_c'],
                    'min_lifetime': base_cfg.generator_params['min_lifetime'],
                    'max_lifetime': base_cfg.generator_params['max_lifetime'],
                    'total_demand': base_cfg.generator_params['total_demand']
                })
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
                import torch as _torch
                agents_tensor = _torch.tensor([[ (a[0], a[1], a[2], obs["time"]) for a in agent_states ]], dtype=_torch.float32, device=device)
                enc_nodes, enc_depot, node_mask = planner._model.encoder(feats)  # type: ignore[attr-defined]
                logits = planner._model.decode(enc_nodes=enc_nodes, enc_depot=enc_depot, node_mask=node_mask, agents_tensor=agents_tensor, nodes=feats.get("nodes"), lateness_lambda=getattr(planner, "lateness_lambda", 0.0))  # type: ignore[attr-defined]
                # labels from teacher targets: map coord to index (0=depot, 1..N nodes)
                nodes = feats["nodes"]
                B, N, _ = nodes.shape
                A = agents_tensor.size(1)
                labels = _torch.zeros((1, A), dtype=_torch.long, device=device)
                # create mapping
                xy_to_idx = { (int(nodes[0, i, 0].item()), int(nodes[0, i, 1].item())): i+1 for i in range(N) }
                depot_xy = (int(feats["depot"][0,0,0].item()), int(feats["depot"][0,0,1].item()))
                for a_idx, q in enumerate(targets):
                    if len(q) == 0:
                        labels[0, a_idx] = 0
                    else:
                        tgt_xy = q[0]
                        labels[0, a_idx] = xy_to_idx.get((int(tgt_xy[0]), int(tgt_xy[1])), 0)
                loss = _torch.nn.functional.cross_entropy(logits.view(1*A, -1), labels.view(-1))
                if opt_planner is not None:
                    opt_planner.zero_grad(); loss.backward(); _torch.nn.utils.clip_grad_norm_(planner._model.parameters(), 1.0); opt_planner.step()  # type: ignore[attr-defined]
                # advance env one step to avoid infinite while in some envs (optional)
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
            if pe % max(1, cfg.planner_epochs_per_cycle // 2) == 0:
                print(f"[PlannerPhase] cycle={cycle} epoch={pe} using gen_version={gv.version_id}")

        planner_ckpt_path = os.path.join(cfg.save_dir, f"planner_cycle_{cycle}.pt")
        if hasattr(planner, "_model") and hasattr(planner._model, "state_dict"):
            torch.save(planner._model.state_dict(), planner_ckpt_path)  # type: ignore[attr-defined]
        print(f"[Save] Planner checkpoint => {planner_ckpt_path}")

        # ---- Generator (adversarial) phase ----
        # Ensure we start evolution from the newest registered generator version
        latest = registry.latest()
        if latest is not None:
            try:
                state = latest.load(device)
                diffusion_model.load_state_dict(state, strict=False)
            except Exception:
                if isinstance(state, dict) and "model" in state:
                    diffusion_model.load_state_dict(state["model"], strict=False)
            diffusion_model.eval()

        for ge in range(1, cfg.generator_epochs_per_cycle + 1):
            # Use current planner (could add evaluation vs older planners here)
            # Evolve the (newest) diffusion model against the planner
            gen_trainer.train(planner, episodes=1, renderer=None, save_path=None, seed=cfg.seed + cycle * 100 + ge)
            if ge % max(1, cfg.generator_epochs_per_cycle // 2) == 0:
                print(f"[GeneratorPhase] cycle={cycle} epoch={ge}")

        gen_ckpt_path = os.path.join(cfg.save_dir, f"generator_cycle_{cycle}.pth")
        torch.save(diffusion_model.state_dict(), gen_ckpt_path)
        registry.add(gen_ckpt_path, metrics={"cycle": cycle})
        if generator_metrics_hook:
            generator_metrics_hook({"cycle": cycle, "ckpt": gen_ckpt_path})
        print(f"[Save] Generator checkpoint => {gen_ckpt_path}")
        print(registry.summary())

    print("=== Coevolution complete ===")
