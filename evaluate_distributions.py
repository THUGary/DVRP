import argparse
import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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


# =============================================================================
# Diffusion Generator Support
# =============================================================================

class DiffusionDistributionGenerator:
    """
    Wrapper to generate demands using a diffusion model checkpoint.
    
    This allows comparing planner performance on diffusion-generated distributions
    versus rule-based distributions (uniform, gaussian, cluster, etc.).
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        map_size: int = 30,
        max_time: int = 100,
        max_end_time: int = 200,
        max_c: int = 5,
        device: str = "cuda",
    ):
        import torch
        from models.generator_model.diffusion_model import DemandDiffusionModel
        from agent.generator.data_utils import prepare_condition, CONDITION_DIM
        from adversarial_v2.utils.demand_converter import DemandConverter
        
        self.checkpoint_path = checkpoint_path
        self.map_size = map_size
        self.max_time = max_time
        self.max_end_time = max_end_time
        self.max_c = max_c
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize diffusion model
        self.model = DemandDiffusionModel(
            condition_dim=CONDITION_DIM,
            data_dim=5,
            time_emb_dim=64,
            num_steps=1000,
        ).to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        
        # Prepare condition
        self.condition = prepare_condition({}).unsqueeze(0).to(self.device)
        
        # Demand converter
        self.converter = DemandConverter(
            map_size=map_size,
            max_time=max_time,
            max_end_time=max_end_time,
            max_c=max_c,
        )
        
        self._label = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    @property
    def label(self) -> str:
        return f"diffusion_{self._label}"
    
    def generate_demands(
        self, 
        num_nodes: int, 
        seed: int = 0,
        static_demands: bool = True,
    ) -> List[tuple]:
        """
        Generate demands using the diffusion model.
        
        Returns:
            List of demand tuples: (x, y, t, c, end_t)
        """
        import torch
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        with torch.no_grad():
            output = self.model.sample(
                condition=self.condition,
                num_demands=num_nodes,
                grid_size=(self.map_size, self.map_size),
            )
        
        if static_demands:
            demands = self.converter.convert_to_static(output)
        else:
            demands = self.converter.convert_to_dynamic(output)
        
        # Convert to tuple format expected by environment
        return [(d.x, d.y, d.t, d.c, d.end_t) for d in demands]


def load_diffusion_generators(
    checkpoint_paths: List[str],
    map_size: int,
    max_time: int,
    max_end_time: int,
    max_c: int,
    device: str = "cuda",
) -> List[DiffusionDistributionGenerator]:
    """Load multiple diffusion generators from checkpoints."""
    generators = []
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"[WARNING] Diffusion checkpoint not found: {path}")
            continue
        try:
            gen = DiffusionDistributionGenerator(
                checkpoint_path=path,
                map_size=map_size,
                max_time=max_time,
                max_end_time=max_end_time,
                max_c=max_c,
                device=device,
            )
            generators.append(gen)
            print(f"[INFO] Loaded diffusion generator: {gen.label}")
        except Exception as e:
            print(f"[WARNING] Failed to load diffusion checkpoint {path}: {e}")
    return generators


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
    diffusion_generators: Optional[List[DiffusionDistributionGenerator]] = None,
):
    """
    Evaluate planners across multiple distributions.
    
    Args:
        cfg: Base configuration
        planner_specs: List of planner specifications to evaluate
        num_runs: Number of evaluation runs per distribution
        static_demands: Whether to use static demand mode
        out_dir: Output directory for results
        max_steps: Maximum episode steps
        diffusion_generators: Optional list of diffusion generators for additional distributions
    """
    import time as time_module
    
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    os.makedirs(out_dir, exist_ok=True)
    
    # Build list of all distributions to evaluate
    all_distributions = list(DISTRIBUTIONS)  # Rule-based distributions
    diffusion_gen_map = {}  # Map distribution name to generator
    
    if diffusion_generators:
        for gen in diffusion_generators:
            all_distributions.append(gen.label)
            diffusion_gen_map[gen.label] = gen
    
    # Calculate total evaluations for progress tracking
    total_evals = len(planner_specs) * len(all_distributions) * num_runs
    completed_evals = 0
    start_time = time_module.time()

    # CUDA Warmup: Initialize CUDA/PyTorch once before any measurements
    # This ensures the first-run CUDA initialization overhead (~1.5s) doesn't
    # affect timing measurements for whichever planner happens to run first.
    print("Performing CUDA warmup...")
    _cuda_warmup()
    print("  CUDA warmup complete")
    
    print(f"\n[INFO] Evaluating {len(planner_specs)} planners on {len(all_distributions)} distributions:")
    print(f"       Rule-based: {DISTRIBUTIONS}")
    if diffusion_gen_map:
        print(f"       Diffusion:  {list(diffusion_gen_map.keys())}")

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

        for dist_idx, dist in enumerate(all_distributions):
            dist_start = time_module.time()
            metrics_list = []
            
            # Check if this is a diffusion distribution
            is_diffusion_dist = dist in diffusion_gen_map

            for seed in range(num_runs):
                local_cfg = copy.deepcopy(cfg)
                
                if is_diffusion_dist:
                    # Use diffusion generator - set generator_type to trigger custom handling
                    diffusion_gen = diffusion_gen_map[dist]
                    # Generate demands using diffusion model
                    num_nodes = local_cfg.generator_params.get("num_nodes", 30)
                    demands = diffusion_gen.generate_demands(
                        num_nodes=num_nodes, 
                        seed=seed,
                        static_demands=static_demands,
                    )
                    # Inject pre-generated demands into config
                    local_cfg.generator_params["distribution"] = "uniform"  # Fallback
                    local_cfg.generator_params["_pregenerated_demands"] = demands
                else:
                    # Standard rule-based distribution
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

    return results, all_distributions


def save_plots_from_results(results: dict[str, dict[str, dict[str, float]]],
                            metrics: list[str],
                            out_dir: str,
                            num_runs: int,
                            dist_names: Optional[List[str]] = None):
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        return

    # Use provided dist_names or infer from results
    if dist_names is None:
        # Get all distribution names from results
        all_dists = set()
        for planner_results in results.values():
            all_dists.update(planner_results.keys())
        dist_names = sorted(all_dists)
    
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

        plt.xticks(x, dist_names, rotation=45, ha='right')
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


def save_episode_length_chart(
    results: dict[str, dict[str, dict[str, float]]], 
    out_dir: str, 
    num_runs: int,
    dist_names: Optional[List[str]] = None,
):
    if not results:
        return
    
    # Use provided dist_names or infer from results
    if dist_names is None:
        all_dists = set()
        for planner_results in results.values():
            all_dists.update(planner_results.keys())
        dist_names = sorted(all_dists)
    
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

    plt.xticks(x, list(dist_names), rotation=45, ha='right')
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
    parser.add_argument(
        "--diffusion-checkpoints", nargs="*", default=[],
        help="List of diffusion generator checkpoint paths for distribution generation. "
             "Supports label=path format. These distributions are evaluated alongside rule-based distributions."
    )
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

    # Load diffusion generators if specified
    diffusion_generators = []
    diffusion_entries = normalize_entries(args.diffusion_checkpoints)
    if diffusion_entries:
        print(f"\n[INFO] Loading {len(diffusion_entries)} diffusion generator(s)...")
        for entry in diffusion_entries:
            if "=" in entry:
                label, path = entry.split("=", 1)
                label = label.strip()
                path = path.strip()
            else:
                path = entry
                label = None  # Will use default label from checkpoint name
            
            if not os.path.exists(path):
                print(f"[WARNING] Diffusion checkpoint not found: {path}")
                continue
            
            try:
                gen = DiffusionDistributionGenerator(
                    checkpoint_path=path,
                    map_size=cfg.width,
                    max_time=cfg.max_time,
                    max_end_time=cfg.max_end_time,
                    max_c=cfg.generator_params.get("max_c", 5),
                    device="cuda",
                )
                if label:
                    gen._label = label  # Override default label
                diffusion_generators.append(gen)
                print(f"  Loaded: {gen.label}")
            except Exception as e:
                print(f"[WARNING] Failed to load diffusion checkpoint {path}: {e}")

    result, all_distributions = evaluate_distributions(
        cfg,
        planner_specs=planner_specs,
        num_runs=args.num_runs,
        static_demands=args.static_demands,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        diffusion_generators=diffusion_generators if diffusion_generators else None,
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
    print(f"Distributions evaluated: {all_distributions}")
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
            for dist in all_distributions:
                stats = result[spec.label].get(dist, {})
                mean_dict = stats.get("mean", {})
                parts = [f"{dist}:"]
                # Print metrics specified in plot_metrics (or default set)
                print_keys = metrics_to_plot if metrics_to_plot else ["failure_flag", "total_distance", "inference_time_total"]
                for key in print_keys:
                    if key in mean_dict:
                        parts.append(f"{key}={mean_dict[key]:.4f}")
                print("    " + " ".join(parts))

    save_plots_from_results(result, metrics_to_plot, args.out_dir, args.num_runs, dist_names=all_distributions)
    if args.static_demands:
        save_episode_length_chart(result, args.out_dir, args.num_runs, dist_names=all_distributions)
