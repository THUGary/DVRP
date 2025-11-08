#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
import torch
import pathlib

# robust project root on path (search upwards for configs.py)
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adversarial.builders import build_env, build_planner, build_diffusion
from adversarial.trainers import DiffusionAdversarialTrainer, AdvConfig
from utils.pygame_renderer import PygameRenderer


def main():
    ap = argparse.ArgumentParser(description="Unified adversarial training: diffusion vs planner")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"]) 
    ap.add_argument("--planner", type=str, default="greedy", choices=["greedy","model"]) 
    ap.add_argument("--planner_ckpt", type=str, default="checkpoints/planner/planner_20_2_200.pt")
    ap.add_argument("--init_diffusion_ckpt", type=str, default="checkpoints/diffusion_model.pth")
    ap.add_argument("--out_ckpt", type=str, default="checkpoints/diffusion_model_adv.pth")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--randomize_depot", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--normalize_reward", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    env, cfg = build_env()
    planner = build_planner(args.planner, cfg, device, args.planner_ckpt if args.planner=="model" else None)
    model, condition = build_diffusion(cfg, device, args.init_diffusion_ckpt)

    adv_cfg = AdvConfig(randomize_depot=args.randomize_depot, lr=args.lr, normalize_reward=args.normalize_reward)
    trainer = DiffusionAdversarialTrainer(env, model, condition, cfg, device, adv_cfg)

    renderer = None
    if args.render:
        try:
            import os as _os
            if not _os.environ.get("DISPLAY"):
                _os.environ["SDL_VIDEODRIVER"] = "dummy"
            renderer = PygameRenderer(cfg.width, cfg.height, cell_size=24, caption="Adversarial Training")
            renderer.init()
        except Exception as e:
            print(f"[Render] init failed: {e}")
            renderer = None

    trainer.train(planner, args.episodes, renderer=renderer, save_path=args.out_ckpt, seed=args.seed)
    if renderer is not None:
        try: renderer.close()
        except Exception: pass

if __name__ == "__main__":
    main()
