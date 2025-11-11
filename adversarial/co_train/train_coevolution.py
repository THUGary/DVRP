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
            context = {
                "env": env,
                "generator_version": gv,
                "rng": rng,
                "opt_planner": opt_planner,
                "diffusion_model": diffusion_model,
                "condition": condition,
                "base_cfg": base_cfg,
                "device": device,
            }
            if planner_update_hook:
                planner_update_hook(planner, context)
                print("rl")
            else:
                # Call the extracted supervised hook
                from adversarial.co_train.supervised_hooks import supervised_planner_hook
                supervised_planner_hook(planner, context)
                print("supervised")
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
