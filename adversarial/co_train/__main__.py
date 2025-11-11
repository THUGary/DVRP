#!/usr/bin/env python3
from __future__ import annotations
import argparse
import torch

from .train_coevolution import coevolution_loop, CoevolutionConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Run co-evolution (planner <-> diffusion) training loop")
    p.add_argument("--num-cycles", type=int, default=5, dest="num_cycles", help="Number of co-training cycles")
    p.add_argument("--planner-epochs", type=int, default=10, dest="planner_epochs", help="Planner epochs per cycle")
    p.add_argument("--generator-epochs", type=int, default=10, dest="generator_epochs", help="Generator (diffusion) epochs per cycle")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Training device")
    p.add_argument("--planner-type", type=str, default="model", choices=["model", "greedy"], help="Planner type to use during co-training")
    p.add_argument("--planner-ckpt-init", type=str, default="checkpoints/planner/planner_20_2_200.pt", help="Initial checkpoint for model planner (if planner-type=model)")
    p.add_argument("--diffusion-ckpt-init", type=str, default="checkpoints/diffusion_model.pth", help="Initial diffusion model checkpoint")
    p.add_argument("--save-dir", type=str, default="checkpoints/coevolution", help="Directory to save coevolution checkpoints")
    p.add_argument("--sample-latest-prob", type=float, default=0.7, dest="sample_latest_prob", help="Probability to sample latest generator in planner phase (rest from history)")
    p.add_argument("--latest-bias", type=float, default=0.7, dest="latest_bias", help="Kept for compatibility; not critical when using sample-latest-prob path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    cfg = CoevolutionConfig(
        num_cycles=args.num_cycles,
        planner_epochs_per_cycle=args.planner_epochs,
        generator_epochs_per_cycle=args.generator_epochs,
        device=device,
        seed=args.seed,
        save_dir=args.save_dir,
        latest_bias=args.latest_bias,
        sample_latest_prob=args.sample_latest_prob,
    )

    coevolution_loop(
        cfg,
        planner_type=args.planner_type,
        planner_ckpt_init=(args.planner_ckpt_init if args.planner_type == "model" else None),
        diffusion_ckpt_init=args.diffusion_ckpt_init,
    )


if __name__ == "__main__":
    main()
