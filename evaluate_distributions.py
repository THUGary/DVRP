import argparse
import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from run_evaluate import run_episode_return_metrics
from configs import get_default_config, Config
from agent.generator.distribution_sets import SUPPORTED_DEMAND_DISTRIBUTIONS

# 支持的分布名称
DISTRIBUTIONS = list(SUPPORTED_DEMAND_DISTRIBUTIONS)


def _cuda_warmup():
    """
    Perform CUDA initialization warmup.
    
    The first CUDA operation in a process incurs a ~1.5s initialization overhead.
    By doing this warmup before any timing measurements, we ensure fair comparison
    between all planners regardless of evaluation order.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Create a small tensor and do a simple operation to trigger CUDA init
            device = torch.device("cuda")
            x = torch.randn(100, 100, device=device)
            _ = torch.matmul(x, x)
            torch.cuda.synchronize()
            del x
    except ImportError:
        pass  # torch not available, skip warmup


def create_output_run_dir(base_dir: str, prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder


def sanitize_cfg(cfg: Config):
    """
    清洗配置，防止 rule_generator 出现 NaN 概率。
    不修改原 cfg，只修复明显导致 /0 或 NaN 的参数。
    """

    g = cfg.generator_params

    if "scale_factor" in g:
        if g["scale_factor"] is None or g["scale_factor"] <= 0:
            g["scale_factor"] = 1.0
    else:
        g["scale_factor"] = 1.0

    if "neighborhood_radius" in g:
        if g["neighborhood_radius"] is None or g["neighborhood_radius"] <= 0:
            g["neighborhood_radius"] = 3
    else:
        g["neighborhood_radius"] = 3

    if "num_centers" in g:
        if g["num_centers"] is None or g["num_centers"] < 1:
            g["num_centers"] = 3
    else:
        g["num_centers"] = 3

    if g.get("distribution") not in DISTRIBUTIONS:
        g["distribution"] = "uniform"

    return cfg


@dataclass
class PlannerSpec:
    label: str
    planner_type: str = "model"
    ckpt_model: Optional[str] = None
    planner_kwargs: Dict[str, Any] = field(default_factory=dict)


def evaluate_distributions(
    cfg: Config,
    planner_specs: list[PlannerSpec],
    num_runs: int = 10,
    *,
    static_demands: bool = False,
    out_dir: str = "outputs/eval",
    max_steps: Optional[int] = None,
):
    import time as time_module
    
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    os.makedirs(out_dir, exist_ok=True)
    
    # Calculate total evaluations for progress tracking
    total_evals = len(planner_specs) * len(DISTRIBUTIONS) * num_runs
    completed_evals = 0
    start_time = time_module.time()

    # CUDA Warmup: Initialize CUDA/PyTorch once before any measurements
    # This ensures the first-run CUDA initialization overhead (~1.5s) doesn't
    # affect timing measurements for whichever planner happens to run first.
    print("Performing CUDA warmup...")
    _cuda_warmup()
    print("  CUDA warmup complete")

    # Note: Model weights are loaded lazily on first use within each planner.
    # The first episode for each model planner will include model loading time,
    # but this is acceptable since we're measuring inference_time separately
    # and the loading overhead is amortized across many runs.

    for spec_idx, spec in enumerate(planner_specs):
        print(f"\n=== Evaluating planner: {spec.label} ({spec.planner_type}) [{spec_idx+1}/{len(planner_specs)}] ===")
        planner_results: dict[str, dict[str, dict[str, float]]] = {}
        
        # For model-based planners, run one warmup episode to load model weights
        # This ensures model loading time doesn't affect inference_time measurements
        is_model_planner = spec.planner_type in ("model", "static", "dynamic")
        if is_model_planner:
            print(f"  [Warmup] Running warmup episode for {spec.label}...")
            warmup_cfg = copy.deepcopy(cfg)
            warmup_cfg.generator_params["distribution"] = DISTRIBUTIONS[0]
            warmup_cfg = sanitize_cfg(warmup_cfg)
            warmup_cfg.seed = 9999  # Use a different seed for warmup
            if spec.ckpt_model:
                if not hasattr(warmup_cfg, 'v2_planner_params'):
                    warmup_cfg.v2_planner_params = {}
                warmup_cfg.v2_planner_params["static_checkpoint"] = spec.ckpt_model
            _ = run_episode_return_metrics(
                warmup_cfg, seed=9999, render=False, fps=0,
                planner=spec.planner_type, static_demands=static_demands,
                planner_kwargs=spec.planner_kwargs, max_steps=max_steps,
            )
            print(f"  [Warmup] Complete")

        for dist_idx, dist in enumerate(DISTRIBUTIONS):
            dist_start = time_module.time()
            metrics_list = []

            for seed in range(num_runs):
                local_cfg = copy.deepcopy(cfg)
                local_cfg.generator_params["distribution"] = dist
                local_cfg = sanitize_cfg(local_cfg)
                local_cfg.seed = seed

                if spec.ckpt_model:
                    # V2Planner checkpoint injection
                    if spec.planner_type in ("model", "static", "dynamic"):
                        if not hasattr(local_cfg, 'v2_planner_params'):
                            local_cfg.v2_planner_params = {}
                        base = os.path.basename(spec.ckpt_model).lower()
                        if "adapter" in base or "adapt" in base:
                            # Key must match create_v2_planner parameter name
                            local_cfg.v2_planner_params["adapter_checkpoint"] = spec.ckpt_model
                        else:
                            local_cfg.v2_planner_params["static_checkpoint"] = spec.ckpt_model

                episode_metrics = run_episode_return_metrics(
                    local_cfg,
                    seed=seed,
                    render=False,
                    fps=0,
                    planner=spec.planner_type,
                    static_demands=static_demands,
                    planner_kwargs=spec.planner_kwargs,
                    max_steps=max_steps,
                )
                metrics_list.append(episode_metrics)
                completed_evals += 1
            
            # Progress update after each distribution
            elapsed = time_module.time() - start_time
            progress_pct = 100.0 * completed_evals / total_evals
            eps = completed_evals / elapsed if elapsed > 0 else 0
            eta = (total_evals - completed_evals) / eps if eps > 0 else 0
            dist_time = time_module.time() - dist_start
            print(f"  -> {dist}: {num_runs} runs in {dist_time:.1f}s | Progress: {progress_pct:.1f}% | ETA: {eta:.0f}s")

            if static_demands:
                selected_keys = [
                    "failure_flag",
                    "total_distance",
                    "episode_steps",
                    "inference_time_avg",
                    "inference_time_total",
                    "inference_time_first",
                    "plan_calls",
                ]
            else:
                selected_keys = [
                    "service_ratio",
                    "total_distance",
                    "inference_time_avg",
                    "inference_time_total",
                    "inference_time_first",
                    "plan_calls",
                ]

            dist_mean: dict[str, float] = {}
            dist_std: dict[str, float] = {}
            for key in selected_keys:
                values = [float(m.get(key, 0.0)) for m in metrics_list]
                dist_mean[key] = float(np.mean(values)) if values else 0.0
                dist_std[key] = float(np.std(values)) if values else 0.0

            planner_results[dist] = {"mean": dist_mean, "std": dist_std}

        results[spec.label] = planner_results

    return results


def save_plots_from_results(results: dict[str, dict[str, dict[str, float]]],
                            metrics: list[str],
                            out_dir: str,
                            num_runs: int):
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        return

    dist_names = DISTRIBUTIONS
    if not dist_names:
        return

    planner_labels = list(results.keys())
    num_planners = len(planner_labels)
    x = np.arange(len(dist_names))
    width = min(0.6, 0.9 / max(num_planners, 1))

    metric_label_map = {"failure_flag": "failure_rate"}

    for metric in metrics:
        display_metric = metric_label_map.get(metric, metric)
        plt.figure(figsize=(max(10, len(dist_names) * 1.2), 6))
        plotted = False
        for i, label in enumerate(planner_labels):
            offsets = x + (i - (num_planners - 1) / 2) * width
            means = []
            stds = []
            for dist in dist_names:
                dist_entry = results[label].get(dist, {})
                mean_val = dist_entry.get("mean", {}).get(metric)
                std_val = dist_entry.get("std", {}).get(metric, 0.0)
                if mean_val is None:
                    means.append(0.0)
                    stds.append(0.0)
                else:
                    means.append(mean_val)
                    stds.append(std_val)
                    plotted = True
            plt.bar(offsets, means, width=width, label=label, yerr=stds, capsize=5, alpha=0.85)

        if not plotted:
            plt.close()
            continue

        plt.xticks(x, dist_names)
        plt.ylabel(display_metric)
        plt.title(f"{display_metric} by distribution (n={num_runs})")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()

        for i, label in enumerate(planner_labels):
            offsets = x + (i - (num_planners - 1) / 2) * width
            values = [results[label].get(dist, {}).get("mean", {}).get(metric) for dist in dist_names]
            available_values = [v for v in values if v is not None]
            max_value = max(available_values) if available_values else 0.0
            text_offset = max_value * 0.01 if max_value > 0 else 0.01
            for xi, offset in enumerate(offsets):
                mean_val = values[xi]
                if mean_val is None:
                    continue
                plt.text(offset, mean_val + text_offset, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8)

            fname = os.path.join(out_dir, f"{display_metric}_by_distribution.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot: {fname}")


def save_episode_length_chart(results: dict[str, dict[str, dict[str, float]]], out_dir: str, num_runs: int):
    if not results:
        return
    dist_names = next(iter(results.values())).keys()
    if not dist_names:
        return
    planner_labels = list(results.keys())
    x = np.arange(len(dist_names))
    width = min(0.6, 0.9 / max(len(planner_labels), 1))

    plt.figure(figsize=(max(10, len(dist_names) * 1.2), 6))
    plotted = False
    for i, label in enumerate(planner_labels):
        offsets = x + (i - (len(planner_labels) - 1) / 2) * width
        means = []
        stds = []
        for dist in dist_names:
            dist_entry = results[label].get(dist, {})
            mean_val = dist_entry.get("mean", {}).get("episode_steps")
            std_val = dist_entry.get("std", {}).get("episode_steps", 0.0)
            if mean_val is None:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(mean_val)
                stds.append(std_val)
                plotted = True
        plt.bar(offsets, means, width=width, label=label, yerr=stds, capsize=5, alpha=0.85)

    if not plotted:
        plt.close()
        return

    plt.xticks(x, list(dist_names))
    plt.ylabel("episode_steps")
    plt.title(f"Episode Length by Distribution (n={num_runs})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    for i, label in enumerate(planner_labels):
        offsets = x + (i - (len(planner_labels) - 1) / 2) * width
        for xi, offset in enumerate(offsets):
            mean_val = results[label].get(list(dist_names)[xi], {}).get("mean", {}).get("episode_steps")
            if mean_val is None:
                continue
            plt.text(offset, mean_val + 1.0, f"{mean_val:.1f}", ha="center", va="bottom", fontsize=8)
    fname = os.path.join(out_dir, "episode_length_by_distribution.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved plot: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate demand distributions with rule-based or model planner")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of episodes per distribution")
    parser.add_argument(
        "--rule-based",
        nargs="*",
        choices=["greedy", "optimize", "exact", "heuristic"],
        default=None,
        action="append",
        help=(
            "Include rule-based planners; pass one or more modes (e.g. --rule-based greedy optimize exact heuristic). "
            "You may repeat the flag; an empty invocation defaults to 'greedy'. "
            "'exact' uses DP to find optimal solution (warns if >12 nodes but still uses DP). "
            "'heuristic' uses Clarke-Wright + local search (good for larger instances)."
        ),
    )
    parser.add_argument(
        "--global-opt",
        nargs="*",
        choices=["hybrid", "cluster_tsp", "sa", "branch_bound"],
        default=None,
        action="append",
        help=(
            "Include global optimization planners; pass one or more modes (e.g. --global-opt hybrid sa). "
            "Available: hybrid, cluster_tsp, sa, branch_bound."
        ),
    )
    parser.add_argument("--ckpt-model", type=str, default=None, help="Checkpoint file for the learned planner (kept for backward compatibility)")
    parser.add_argument("--model-checkpoints", nargs="*", default=[],
                        help="List of checkpoint paths for models to compare. Supports label=path. Brackets/comma-separated lists are normalized.")
    parser.add_argument("--use-hungarian", action="store_true",
                        help="Use Hungarian algorithm for model planner's global assignment (instead of greedy decoding)")
    parser.add_argument("--total-demand", type=int, default=None,
                        help="Override total demand capacity (upper limit of sum of all demands)")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Override number of demand nodes")
    parser.add_argument("--map-size", type=int, default=None, help="Override map size (square map: map_size × map_size)")
    parser.add_argument("--map-wid", type=int, default=None, help="Override map width (deprecated, use --map-size)")
    parser.add_argument("--map-hei", type=int, default=None, help="Override map height (deprecated, use --map-size)")
    parser.add_argument("--num-agents", type=int, default=None, help="Override number of agents used for evaluation")
    parser.add_argument("--capacity", type=int, default=None,
                        help="Override vehicle capacity. For POMO model, use capacity~30 for 2 agents to match training normalization")
    parser.add_argument("--max-c", type=int, default=None,
                        help="Override max demand per node. Higher values make capacity constraint more meaningful (default=5)")
    parser.add_argument("--static-demands", action="store_true", help="Use static demand release (all demands at t=0)")
    parser.add_argument("--static-max-end", type=int, default=None,
                        help="When --static-demands is set, override the max_end_time (fail if exceeded).")
    parser.add_argument("--out-dir", type=str, default="outputs/eval", help="Directory where plots will be written")
    parser.add_argument("--plot-metrics", type=str, default="service_ratio,total_distance",
                        help="Comma-separated metrics to visualize (defaults to service_ratio,total_distance)")
    parser.add_argument("--pomo-size", type=int, default=20,
                        help="POMO parallel rollouts for model inference (higher=better but slower, default=20)")
    parser.add_argument("--aug-factor", type=int, default=8, choices=[1, 8],
                        help="Data augmentation factor for POMO inference (1 or 8, default=8)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum episode steps (default: no limit)")
    args = parser.parse_args()

    cfg = get_default_config()
    evaluation_root = args.out_dir or "outputs/eval"
    eval_run_dir = create_output_run_dir(evaluation_root, "eval")
    args.out_dir = eval_run_dir
    print(f"[EVAL] Output directory: {eval_run_dir}")
    planner_specs: list[PlannerSpec] = []

    rule_modes: list[str] = []
    if args.rule_based:
        for entry in args.rule_based:
            if entry:
                rule_modes.extend(entry)
            else:
                rule_modes.append("greedy")
    elif args.rule_based == []:
        rule_modes.append("greedy")
    seen_modes: set[str] = set()
    for mode in rule_modes:
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        planner_specs.append(
            PlannerSpec(
                label=f"rule_{mode}",
                planner_type="rule",
                planner_kwargs={"mode": mode},
            )
        )

    # Global optimization planners
    global_opt_modes: list[str] = []
    if args.global_opt:
        for entry in args.global_opt:
            if entry:
                global_opt_modes.extend(entry)
            else:
                global_opt_modes.append("hybrid")
    seen_global_modes: set[str] = set()
    for mode in global_opt_modes:
        if mode in seen_global_modes:
            continue
        seen_global_modes.add(mode)
        planner_specs.append(
            PlannerSpec(
                label=f"global_{mode}",
                planner_type="global_opt",
                planner_kwargs={"mode": mode, "time_limit": 0.1},
            )
        )

    def normalize_entries(raw_entries: list[str]) -> list[str]:
        normalized: list[str] = []
        for entry in raw_entries:
            entry = entry.strip()
            if not entry:
                continue
            if entry.startswith("[") and entry.endswith("]"):
                entry = entry[1:-1]
            for part in entry.split(','):
                part = part.strip()
                if part:
                    normalized.append(part)
        return normalized

    checkpoint_entries = normalize_entries(args.model_checkpoints)
    if args.ckpt_model:
        checkpoint_entries.append(args.ckpt_model)

    # Build planner_kwargs for model planners
    model_planner_kwargs = {}
    if args.use_hungarian:
        model_planner_kwargs["use_hungarian"] = True

    for entry in checkpoint_entries:
        if not entry:
            continue
        if "=" in entry:
            label, path = entry.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = entry
            label = os.path.splitext(os.path.basename(path))[0] or "model"
        
        # Determine planner type based on checkpoint name and --static-demands
        # For static demands with a static checkpoint, use "static" mode
        # For adapter checkpoints or dynamic scenarios, use "dynamic" mode
        base_name = os.path.basename(path).lower()
        if "adapter" in base_name or "adapt" in base_name:
            ptype = "dynamic"
        elif args.static_demands:
            # Static demands + static checkpoint => use static mode
            ptype = "static"
        else:
            # Default to dynamic for dynamic VRP scenarios
            ptype = "model"
        
        planner_specs.append(PlannerSpec(
            label=label, 
            planner_type=ptype, 
            ckpt_model=path,
            planner_kwargs=model_planner_kwargs.copy()
        ))

    if not planner_specs:
        planner_specs.append(PlannerSpec(label="model", planner_type="model"))

    # Handle map size: prefer --map-size over --map-wid/--map-hei
    if args.map_size is not None and args.map_size > 0:
        cfg.width = args.map_size
        cfg.height = args.map_size
    else:
        if args.map_wid is not None and args.map_wid > 0:
            cfg.width = args.map_wid
        if args.map_hei is not None and args.map_hei > 0:
            cfg.height = args.map_hei
    if args.num_agents is not None and args.num_agents > 0:
        cfg.num_agents = int(args.num_agents)
    if args.capacity is not None and args.capacity > 0:
        cfg.capacity = int(args.capacity)
    if args.total_demand is not None and args.total_demand > 0:
        cfg.generator_params = dict(cfg.generator_params)
        cfg.generator_params["total_demand"] = args.total_demand
    if args.num_nodes is not None and args.num_nodes > 0:
        cfg.generator_params = dict(cfg.generator_params)
        cfg.generator_params["num_nodes"] = args.num_nodes
    if args.max_c is not None and args.max_c > 0:
        cfg.generator_params = dict(cfg.generator_params)
        cfg.generator_params["max_c"] = args.max_c
    if args.static_demands:
        if args.static_max_end is not None and args.static_max_end > 0:
            cfg.max_end_time = args.static_max_end
        else:
            cfg.max_end_time = max(cfg.max_time * 10, cfg.max_time + 200)

    # Set POMO parameters in v2_planner_params
    if not hasattr(cfg, 'v2_planner_params'):
        cfg.v2_planner_params = {}
    cfg.v2_planner_params["pomo_size"] = args.pomo_size
    cfg.v2_planner_params["aug_factor"] = args.aug_factor

    result = evaluate_distributions(
        cfg,
        planner_specs=planner_specs,
        num_runs=args.num_runs,
        static_demands=args.static_demands,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
    )

    # Parse metrics to plot before printing results
    metrics_to_plot = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    if args.static_demands:
        normalized: list[str] = []
        for metric in metrics_to_plot:
            if metric == "service_ratio":
                normalized.append("failure_flag")
            else:
                normalized.append(metric)
        if "failure_flag" not in normalized:
            normalized.insert(0, "failure_flag")
        if "total_distance" not in normalized:
            normalized.append("total_distance")
        metrics_to_plot = normalized

    print("\n==== Final Distribution Evaluation Results ====")
    for spec in planner_specs:
        msg = f"Method: {spec.label} ({spec.planner_type})"
        if spec.planner_type in ("model", "static", "dynamic"):
            path = spec.ckpt_model or "<no checkpoint provided>"
            exists = os.path.exists(spec.ckpt_model) if spec.ckpt_model else False
            status = "found" if exists else "missing"
            msg += f" -> checkpoint {status}: {path}"
        print(msg)
        # Print per-distribution metrics for this planner
        if spec.label in result:
            for dist, stats in result[spec.label].items():
                mean_dict = stats.get("mean", {})
                parts = [f"{dist}:"]
                # Print metrics specified in plot_metrics (or default set)
                print_keys = metrics_to_plot if metrics_to_plot else ["failure_flag", "total_distance", "inference_time_total"]
                for key in print_keys:
                    if key in mean_dict:
                        parts.append(f"{key}={mean_dict[key]:.4f}")
                print("    " + " ".join(parts))

    save_plots_from_results(result, metrics_to_plot, args.out_dir, args.num_runs)
    if args.static_demands:
        save_episode_length_chart(result, args.out_dir, args.num_runs)
