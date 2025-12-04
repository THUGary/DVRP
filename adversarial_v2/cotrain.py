#!/usr/bin/env python3
"""
Co-evolution Training Entry Point

Arguments are parsed here and stored in dataclass objects (CoevolutionConfig, EnvironmentConfig).
These config objects are then passed to other modules.
"""
from __future__ import annotations
import argparse
import sys

from adversarial_v2.config import CoevolutionConfig, EnvironmentConfig
from adversarial_v2.coevolution import coevolution_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Co-evolution training for V2Planner and Diffusion Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Main settings
    parser.add_argument(
        "--mode", type=str, default="static", choices=["static", "dynamic"],
        help="Training mode: 'static' for all demands at t=0, 'dynamic' for time-varying demands"
    )
    parser.add_argument(
        "--num-cycles", type=int, default=10,
        help="Number of co-evolution cycles"
    )
    parser.add_argument(
        "--planner-epochs", type=int, default=5,
        help="Planner training epochs per cycle"
    )
    parser.add_argument(
        "--first-cycle-planner-epochs", type=int, default=None,
        help="Planner training epochs for the first cycle (overrides --planner-epochs for cycle 1). "
             "Useful for longer initial training when starting from scratch."
    )
    parser.add_argument(
        "--generator-epochs", type=int, default=5,
        help="Generator training epochs per cycle"
    )
    
    # Planner early stopping
    parser.add_argument(
        "--planner-early-stop-patience", type=int, default=3,
        help="Stop planner training if score doesn't improve for this many epochs (0 to disable)"
    )
    parser.add_argument(
        "--planner-early-stop-threshold", type=float, default=0.01,
        help="Minimum score improvement threshold for early stopping"
    )
    
    # Generator early stopping
    parser.add_argument(
        "--generator-early-stop-patience", type=int, default=0,
        help="Stop generator training if gen_reward doesn't improve for this many epochs (0 to disable)"
    )
    parser.add_argument(
        "--generator-early-stop-threshold", type=float, default=0.1,
        help="Minimum gen_reward improvement threshold for generator early stopping"
    )
    
    # Version sampling
    parser.add_argument(
        "--version-policy", type=str, default="latest_biased",
        choices=["uniform", "latest_biased", "all"],
        help="Policy for sampling generator versions during planner training"
    )
    parser.add_argument(
        "--latest-bias", type=float, default=0.7,
        help="Probability of sampling latest version when using latest_biased policy"
    )
    
    # Batch settings
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for training (reduce for limited GPU memory)"
    )
    parser.add_argument(
        "--pomo-size", type=int, default=20,
        help="POMO parallel rollouts (reduce for limited GPU memory, default=20)"
    )
    parser.add_argument(
        "--episodes-per-epoch", type=int, default=1000,
        help="Episodes per training epoch"
    )
    
    # Environment settings
    parser.add_argument(
        "--map-size", type=int, default=20,
        help="Side length of the square map (map is map_size × map_size)"
    )
    parser.add_argument("--num-agents", type=int, default=5, help="Number of vehicles")
    parser.add_argument("--capacity", type=int, default=30, help="Vehicle capacity (fixed at 30 = DEMAND_NORM)")
    parser.add_argument("--max-time", type=int, default=100, help="Max simulation time")
    parser.add_argument("--max-end-time", type=int, default=200, help="Max end time for demands (deadline)")
    parser.add_argument(
        "--num-nodes", type=int, default=50,
        help="Number of demand nodes (actual count for tensor shapes). "
             "Distinct from --total-demand which is capacity upper limit."
    )
    parser.add_argument(
        "--total-demand", type=int, default=150,
        help="Upper limit of sum of all customer demands (NOT node count!). "
             "This is the capacity constraint, not the number of nodes."
    )
    parser.add_argument("--max-c", type=int, default=5, help="Max demand per node (demands are 1 to max_c)")
    parser.add_argument("--min-lifetime", type=int, default=10, help="Min demand lifetime")
    parser.add_argument("--max-lifetime", type=int, default=50, help="Max demand lifetime")
    parser.add_argument("--randomize-depot", action="store_true", default=False, help="Randomize depot location")
    
    # Checkpoints (optional, for loading pretrained models)
    parser.add_argument(
        "--planner-checkpoint", type=str, default=None,
        help="Path to pretrained planner model checkpoint"
    )
    parser.add_argument(
        "--generator-checkpoint", type=str, default=None,
        help="Path to pretrained diffusion generator checkpoint"
    )
    
    # Training settings
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/cotrain",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from"
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # =================================================================
    # Build config dataclasses from CLI arguments (single entry point)
    # =================================================================
    
    # Environment config
    env_config = EnvironmentConfig(
        map_size=args.map_size,
        num_agents=args.num_agents,
        capacity=args.capacity,
        max_time=args.max_time,
        max_end_time=args.max_end_time,
        num_nodes=args.num_nodes,
        total_demand=args.total_demand,
        max_c=args.max_c,
        min_lifetime=args.min_lifetime,
        max_lifetime=args.max_lifetime,
        randomize_depot=args.randomize_depot,
    )
    
    # Main coevolution config (contains env_config)
    config = CoevolutionConfig(
        mode=args.mode,
        num_cycles=args.num_cycles,
        planner_epochs_per_cycle=args.planner_epochs,
        generator_epochs_per_cycle=args.generator_epochs,
        first_cycle_planner_epochs=args.first_cycle_planner_epochs,
        planner_early_stop_patience=args.planner_early_stop_patience,
        planner_early_stop_threshold=args.planner_early_stop_threshold,
        generator_early_stop_patience=args.generator_early_stop_patience,
        generator_early_stop_threshold=args.generator_early_stop_threshold,
        batch_size=args.batch_size,
        pomo_size=args.pomo_size,
        episodes_per_epoch=args.episodes_per_epoch,
        version_sample_policy=args.version_policy,
        latest_bias=args.latest_bias,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        planner_checkpoint=args.planner_checkpoint,
        generator_checkpoint=args.generator_checkpoint,
        env=env_config,
    )
    
    # =================================================================
    # Run training (pass config object, not individual args)
    # =================================================================
    history = coevolution_loop(config, resume_from=args.resume)
    
    # Print summary
    print("\nTraining Summary:")
    if history["planner_scores"]:
        print(f"  Final planner score: {history['planner_scores'][-1]:.4f}")
    if history["generator_rewards"]:
        print(f"  Final generator reward: {history['generator_rewards'][-1]:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
