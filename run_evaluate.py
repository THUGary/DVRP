from __future__ import annotations
import argparse
import random
import os
import copy
from typing import List, Tuple, Dict, Any, Optional

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.controller import RuleBasedController
from utils.pygame_renderer import PygameRenderer
from utils.state_manager import PlanningState, update_planning_state
from agent.generator.base import BaseDemandGenerator
from agent.planner.base import BasePlanner
from agent.planner import RuleBasedPlanner
from agent.planner import FastReactiveInserter
from agent.planner import RepairBasedStabilityOptimizer
from agent.planner import DistributedCooperativePlanner
from agent.planner import V2Planner, create_v2_planner
from datetime import datetime

import time
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agent.generator.distribution_sets import SUPPORTED_DEMAND_DISTRIBUTIONS
from agent.generator.factory import build_rule_based_generator

_PLANNER_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...], Tuple[Tuple[str, Any], ...]] , BasePlanner] = {}


def _make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(v) for v in value)
    return value


def _planner_cache_key(
    planner_type: str,
    cfg: Config,
    v2_params: Dict[str, Any],
    planner_kwargs: Optional[Dict[str, Any]],
) -> Tuple[str, Tuple[Tuple[str, Any], ...], Tuple[Tuple[str, Any], ...]]:
    param_tuple = tuple(sorted((k, _make_hashable(v)) for k, v in v2_params.items()))
    kwargs_tuple = tuple(sorted((k, _make_hashable(v)) for k, v in (planner_kwargs or {}).items()))
    size_tuple = (
        ("width", cfg.width),
        ("height", cfg.height),
        ("capacity", cfg.capacity),
    )
    combined = param_tuple + size_tuple
    return (planner_type, combined, kwargs_tuple)

# =========================================================
# EvaluationTracker (from old run.py)
# =========================================================
class EvaluationTracker:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.last_positions = [None] * num_agents
        self.total_distance = 0.0
        self.agent_distance = [0.0] * num_agents
        self.paths = [[] for _ in range(num_agents)]
        self.total_requests = 0
        self.served_requests = 0
        self.expired_requests = 0
        self.response_times: List[int] = []
        self.served_times: List[int] = []
        # key tracks (x, y, release_t, capacity, end_t) so static duplicates remain unique
        self._request_registry: Dict[Tuple[int, ...], int] = {}
        self.load_history: List[List[int]] = [[] for _ in range(num_agents)]

    @staticmethod
    def _tuple_key_from_d(d):
        if d is None:
            return None
        if isinstance(d, (list, tuple)):
            try:
                if len(d) >= 5:
                    return (int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]))
                if len(d) >= 3:
                    return (int(d[0]), int(d[1]), int(d[2]))
            except Exception:
                return None
        return None

    def register_new_demands(self, new_demands, current_time: int):
        if new_demands is None:
            return
        if isinstance(new_demands, (list, tuple)) and len(new_demands) >= 5 and not any(isinstance(x, (list, tuple)) for x in new_demands[:1]):
            new_demands = [new_demands]
        for d in new_demands:
            try:
                key = self._tuple_key_from_d(d)
                if key is None:
                    continue
                if key not in self._request_registry:
                    release_time = int(current_time)
                    if isinstance(d, (list, tuple)) and len(d) >= 3:
                        try:
                            release_time = int(d[2])
                        except Exception:
                            pass
                    self._request_registry[key] = release_time
                    self.total_requests += 1
            except Exception:
                continue

    def record_path_and_distance(self, agent_states, time_step: int, env):
        for i, (x, y, load) in enumerate(agent_states):
            self.paths[i].append((int(time_step), int(x), int(y), int(load)))
            try:
                full = env._full_capacity()
                carried = int(max(0, full - int(load)))
            except Exception:
                try:
                    carried = int(load)
                except Exception:
                    carried = 0
            self.load_history[i].append(carried)
            last = self.last_positions[i]
            if last is not None:
                lx, ly = last
                d = abs(int(x) - int(lx)) + abs(int(y) - int(ly))
                self.total_distance += float(d)
                self.agent_distance[i] += float(d)
            self.last_positions[i] = (int(x), int(y))

    def record_served_by_tuple(self, d_tuple, served_time: int):
        x, y, t, c, end_t = d_tuple
        key = self._tuple_key_from_d(d_tuple)
        req_time = None
        if key is not None:
            req_time = self._request_registry.pop(key, None)
        if req_time is None:
            fallback = (int(x), int(y), int(t))
            req_time = self._request_registry.pop(fallback, None)
        if req_time is not None:
            rt = int(served_time) - int(req_time)
            if rt >= 0:
                self.response_times.append(rt)
        self.served_requests += 1
        try:
            self.served_times.append(int(served_time))
        except Exception:
            pass

    def record_expired(self, d_tuple):
        try:
            x, y, t, c, end_t = d_tuple
            trip_key = self._tuple_key_from_d(d_tuple)
            fallback_key = (int(x), int(y), int(t))
            if trip_key in self._request_registry:
                self._request_registry.pop(trip_key, None)
            elif fallback_key in self._request_registry:
                self._request_registry.pop(fallback_key, None)
            else:
                keys_to_remove = [k for k in list(self._request_registry.keys()) if k[0] == int(x) and k[1] == int(y)]
                for k in keys_to_remove:
                    self._request_registry.pop(k, None)
        except Exception:
            coord = self._tuple_key_from_d(d_tuple)
            if coord is not None:
                keys_to_remove = [k for k in list(self._request_registry.keys()) if k[0] == coord[0] and k[1] == coord[1]]
                for k in keys_to_remove:
                    self._request_registry.pop(k, None)
        self.expired_requests += 1

    def finalize(self) -> Dict[str, Any]:
        vehicles_used = sum(1 for d in self.agent_distance if d > 0)
        nz = [d for d in self.agent_distance if d > 0]
        avg_route_length = sum(nz) / len(nz) if nz else 0.0
        route_balance_std = float(np.std(nz)) if nz else 0.0
        avg_response_time = float(np.mean(self.response_times)) if len(self.response_times) > 0 else 0.0
        service_ratio = float(self.served_requests) / float(self.total_requests) if self.total_requests > 0 else 0.0
        return {
            "total_distance": float(self.total_distance),
            "agent_distance": list(self.agent_distance),
            "avg_route_length": float(avg_route_length),
            "route_balance_std": float(route_balance_std),
            "vehicles_used": int(vehicles_used),
            "total_requests": int(self.total_requests),
            "served_requests": int(self.served_requests),
            "expired_requests": int(self.expired_requests),
            "service_ratio": float(service_ratio),
            "avg_response_time": float(avg_response_time),
            "paths": list(self.paths),
            "loads_per_step": list(self.load_history),
            "served_times": list(self.served_times),
        }

# =========================================================
def convert_all(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {convert_all(k): convert_all(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_all(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_all(v) for v in obj]
    if isinstance(obj, (_np.integer, _np.int32, _np.int64)):
        return int(obj)
    if isinstance(obj, (_np.floating, _np.float32, _np.float64)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def create_output_run_dir(root: str, prefix: str = "eval") -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(root, f"{prefix}_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder

# =========================================================
def build_env(
    cfg: Config,
    planner_type: str,
    *,
    static_demands: bool = False,
    planner_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[GridEnvironment, BaseDemandGenerator, BasePlanner, RuleBasedController]:
    planner_kwargs = planner_kwargs or {}
    resolved_max_end_time = int(getattr(cfg, "max_end_time", cfg.max_time * 2))
    if cfg.generator_type == "net":
        from agent.generator.net_generator import NetDemandGenerator as GenClass

        gen = GenClass(cfg.width, cfg.height, **cfg.generator_params)
        env_static_demands = static_demands
    else:
        gen = build_rule_based_generator(
            cfg.width,
            cfg.height,
            cfg.generator_params,
            depot=cfg.depot,
            static_demands=static_demands,
            max_end_time=resolved_max_end_time,
        )
        # Even though static wrapper handles demand generation, environment needs
        # static_demands=True for early termination logic (terminate when all
        # demands served and all agents at depot, regardless of max_time)
        env_static_demands = static_demands
    env = GridEnvironment(
        width=cfg.width,
        height=cfg.height,
        num_agents=cfg.num_agents,
        capacity=cfg.capacity,
        depot=cfg.depot,
        generator=gen,
        max_time=cfg.max_time,
        expiry_penalty_scale=float(getattr(cfg, "expiry_penalty_scale", 5.0)),
        switch_penalty_scale=float(getattr(cfg, "switch_penalty_scale", 0.01)),
        capacity_reward_scale=float(getattr(cfg, "capacity_reward_scale", 10.0)),
        exploration_history_n=int(getattr(cfg, "exploration_history_n", 0)),
        exploration_penalty_scale=float(getattr(cfg, "exploration_penalty_scale", 0.0)),
        wait_penalty_scale=float(getattr(cfg, "wait_penalty_scale", 0.001)),
        depot_return_bonus_scale=float(getattr(cfg, "depot_return_bonus_scale", 0.0)),
        max_end_time=resolved_max_end_time,
        include_service_time=bool(getattr(cfg, "include_service_time", False)),
        static_demands=env_static_demands,
    )
    env.num_agents = cfg.num_agents

    if planner_type in ("greedy", "rule", "optimize"):
        mode = planner_kwargs.get("mode") if planner_kwargs else None
        if mode is None:
            mode = "optimize" if planner_type == "optimize" else "greedy"
        planner_params = dict(cfg.planner_params)
        planner_params.pop("mode", None)
        planner = RuleBasedPlanner(full_capacity=cfg.capacity, mode=mode, **planner_params)
    elif planner_type == "fri":
        planner = FastReactiveInserter()
    elif planner_type == "rbso":
        planner = RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
    elif planner_type == "dcp":
        planner = DistributedCooperativePlanner(auction_rounds=5, bid_strategy='time_urgency')
    elif planner_type in ("global", "global_opt", "global_optimizer"):
        # Global optimization planner with configurable mode
        from agent.planner import GlobalOptimizationPlanner
        opt_mode = planner_kwargs.get("mode", "hybrid") if planner_kwargs else "hybrid"
        time_limit = planner_kwargs.get("time_limit", 0.05) if planner_kwargs else 0.05
        planner = GlobalOptimizationPlanner(
            full_capacity=cfg.capacity,
            mode=opt_mode,
            time_limit=time_limit,
        )
    elif planner_type in ("model", "static", "dynamic"):
        # Use V2Planner for model/static/dynamic
        # NOTE: V2Planner now uses standardized normalization from configs.py
        # - DEMAND_NORM is the normalization constant (= vehicle capacity = 30)
        # - cfg.capacity should always be 30 (fixed)
        # - Model sees vehicle_capacity = cfg.capacity / DEMAND_NORM = 30/30 = 1.0
        v2_params = dict(getattr(cfg, 'v2_planner_params', {}))
        # Map "model" to "dynamic" mode
        mode = "dynamic" if planner_type in ("model", "dynamic") else "static"
        cache_key = _planner_cache_key(planner_type, cfg, v2_params, planner_kwargs)
        planner = _PLANNER_CACHE.get(cache_key)
        if planner is None:
            # Remove deprecated demand_norm from params - now handled by configs.py constants
            v2_params.pop('demand_norm', None)
            planner = create_v2_planner(
                mode=mode,
                grid_width=cfg.width,
                grid_height=cfg.height,
                full_capacity=cfg.capacity,
                **v2_params,
            )
            _PLANNER_CACHE[cache_key] = planner
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
    controller = RuleBasedController(**cfg.controller_params)
    return env, gen, planner, controller

# =========================================================
# run_episode + EvaluationTracker metrics (融合新旧)
# =========================================================
def run_episode_return_metrics(
    cfg: Config,
    seed: int = 0,
    render: bool = False,
    fps: int = 10,
    planner: str = "greedy",
    *,
    static_demands: bool = False,
    planner_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))
    cfg = copy.deepcopy(cfg)
    cfg.depot = depot
    cfg.generator_params = {**cfg.generator_params, "depot": depot}
    planner_type = planner
    env, gen, planner_impl, controller = build_env(
        cfg,
        planner_type,
        static_demands=static_demands,
        planner_kwargs=planner_kwargs,
    )
    obs = env.reset(seed)
    total_reward = 0.0
    done = False
    step = 0
    renderer = None
    planning_state = PlanningState()
    planning_state.reset(cfg.num_agents)
    tracker = EvaluationTracker(cfg.num_agents)
    tracker.register_new_demands(obs.get("demands", []), obs.get("time", 0))
    prev_demands = list(obs["demands"])
    total_inference_time = 0.0
    first_inference_time = 0.0  # Track first inference separately
    plan_calls = 0

    if render:
        renderer = PygameRenderer(cfg.width, cfg.height)
        renderer.init()

    # Static mode optimization: plan once and execute
    # For static problems, we only need to plan once after all demands are revealed
    # Only certain planners support one-shot planning (model, exact, heuristic)
    # greedy/optimize are step-by-step planners that need to replan each step
    static_plan_cached = None
    planner_mode = (planner_kwargs or {}).get("mode", "greedy") if planner_type == "rule" else None
    planners_supporting_static_cache = {"model", "static", "dynamic"}
    rule_modes_supporting_static_cache = {"exact", "heuristic"}
    use_static_plan_cache = static_demands and (
        planner_type in planners_supporting_static_cache or
        (planner_type == "rule" and planner_mode in rule_modes_supporting_static_cache)
    )

    while not done:
        current_demands = obs["demands"]

        # ===== 新的 total_requests 修复 =====
        new_demands = [d for d in current_demands if tracker._tuple_key_from_d(d) not in tracker._request_registry]
        if new_demands:
            tracker.register_new_demands(new_demands, obs.get("time", 0))

        agent_states = obs["agent_states"]
        update_planning_state(
            planning_state=planning_state,
            agent_states=agent_states,
            new_demands=new_demands,
            obs_demands=current_demands,
            depot=obs["depot"],  # 传入 depot 以便在清理时保留 depot 目标
        )

        agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
        plan_horizon = 1 if planner_type not in ("model", "dynamic") else max(1, int(getattr(cfg, 'v2_planner_params', {}).get("time_plan", 1)))
        
        # Static mode: only plan once when demands are available and cache the result
        if use_static_plan_cache and static_plan_cached is not None:
            # Use cached plan - update targets based on served demands
            # Get current demand positions to check if targets are still needed
            current_demand_positions = set((d[0], d[1]) for d in current_demands)
            # Build demand capacity map for capacity checking
            demand_capacity_map = {(d[0], d[1]): d[3] for d in current_demands}
            depot = obs["depot"]
            
            targets = []
            for i, agent in enumerate(agents):
                agent_targets = static_plan_cached[i]
                agent_x, agent_y, agent_cap = agent_states[i]
                
                # Remove targets from the front that have been served (no longer in demands)
                # OR that the agent has already visited and are no longer needed
                while agent_targets:
                    target_pos = agent_targets[0]
                    # Keep depot targets, only remove demand targets that have been served
                    if target_pos == depot:
                        break
                    # If demand at this position has been served (not in current demands),
                    # remove it from the target queue
                    if target_pos not in current_demand_positions:
                        agent_targets.popleft()
                    else:
                        break
                
                # If all targets visited, return to depot
                if not agent_targets:
                    agent_targets.append(depot)
                
                # CAPACITY CHECK: If agent is at target but lacks capacity,
                # temporarily redirect to depot to refill before continuing
                if agent_targets and agent_targets[0] != depot:
                    next_target = agent_targets[0]
                    demand_cap = demand_capacity_map.get(next_target, 0)
                    at_target = ((agent_x, agent_y) == next_target)
                    # If at target but can't serve, or approaching but won't have capacity
                    if at_target and agent_cap < demand_cap:
                        # Insert depot at front to go refill
                        agent_targets.appendleft(depot)
                
                targets.append(agent_targets)
        else:
            # Regular planning
            plan_start = time.perf_counter()
            targets = planner_impl.plan(
                observations=obs["demands"],
                agent_states=agents,
                depot=obs["depot"],
                t=obs["time"],
                horizon=plan_horizon,
                current_plans=planning_state.current_plans,
                global_nodes=planning_state.global_nodes.nodes,
                serve_mark=planning_state.global_nodes.serve_mark,
                unserved_count=planning_state.get_unserved_count(),
            )
            elapsed = time.perf_counter() - plan_start
            plan_calls += 1
            total_inference_time += elapsed
            # Track first inference time (meaningful for static problems)
            if plan_calls == 1:
                first_inference_time = elapsed
            
            # Cache the plan for static mode (after demands are revealed)
            if use_static_plan_cache and current_demands and static_plan_cached is None:
                # Deep copy the targets to avoid mutation
                from collections import deque
                static_plan_cached = [deque(list(t)) for t in targets]
        
        planning_state.update_plans(targets)

        actions = [controller.act((x, y), targets[i]) for i, (x, y, s) in enumerate(agent_states)]
        obs_after, reward, done, info = env.step(actions)
        current_time = obs_after.get("time", 0)

        disappeared = [d for d in prev_demands if d not in obs_after["demands"]]
        for d in disappeared:
            x, y, t, c, end_t = d
            if int(end_t) >= int(current_time):
                tracker.record_served_by_tuple(d, served_time=current_time)
            else:
                tracker.record_expired(d)

        tracker.record_path_and_distance(obs_after["agent_states"], current_time, env)
        prev_demands = list(obs_after["demands"])
        obs = obs_after
        total_reward += reward
        step += 1

        # Early termination if max_steps is reached
        if max_steps is not None and step >= max_steps:
            break

    if renderer:
        renderer.close()

    metrics = tracker.finalize()
    total_requests = int(metrics.get("total_requests", 0))
    served_requests = int(metrics.get("served_requests", 0))
    failure_flag = 1.0 if total_requests > served_requests else 0.0
    metrics["failure_flag"] = float(failure_flag)
    metrics["episode_steps"] = step
    metrics["inference_time_avg"] = float(total_inference_time / plan_calls) if plan_calls else 0.0
    metrics["inference_time_total"] = float(total_inference_time)
    metrics["inference_time_first"] = float(first_inference_time)  # First call time (meaningful for static)
    metrics["plan_calls"] = int(plan_calls)
    return metrics

# =========================================================
# evaluate_distributions (迁移旧 run.py)
# =========================================================
def evaluate_distributions(cfg: Config, planner_choice: str, num_runs: int = 10, out_dir: str = "outputs/eval"):
    os.makedirs(out_dir, exist_ok=True)
    distributions = list(SUPPORTED_DEMAND_DISTRIBUTIONS)
    metric_names = ["total_distance", "avg_response_time", "service_ratio", "avg_route_length", "vehicles_used"]
    aggregated = {dist: {m: [] for m in metric_names} for dist in distributions}
    base_seed = 1000
    for dist in distributions:
        print(f"=== Evaluating distribution: {dist} ===")
        for run_idx in range(num_runs):
            seed = base_seed + run_idx
            local_cfg = copy.deepcopy(cfg)
            local_cfg.generator_type = local_cfg.generator_type if hasattr(local_cfg, "generator_type") else "rule"
            local_cfg.generator_params = dict(local_cfg.generator_params)
            local_cfg.generator_params["distribution"] = dist
            metrics = run_episode_return_metrics(local_cfg, seed=seed, render=False, fps=0, planner=planner_choice)
            for m in metric_names:
                val = metrics.get(m, None)
                aggregated[dist][m].append(val if val is not None else 0.0)
            print(f"  run {run_idx+1}/{num_runs} seed {seed}: service_ratio={metrics.get('service_ratio',0):.3f}, total_distance={metrics.get('total_distance',0):.1f}")

        for m in metric_names:
            arr = np.array(aggregated[dist][m], dtype=float)
            aggregated[dist][m] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        # 保存图像
        x = np.arange(len(distributions))
        width = 0.6
        for m in metric_names:
            means = [aggregated[d][m]["mean"] for d in distributions]
            stds = [aggregated[d][m]["std"] for d in distributions]
            plt.figure(figsize=(10, 6))
            bars = plt.bar(x, means, yerr=stds, capsize=5, width=width)
            plt.xticks(x, distributions)
            plt.ylabel(m)
            plt.title(f"{m} by distribution (mean ± std, n={num_runs})")
            for xi, val in enumerate(means):
                plt.text(xi, val + (max(means) * 0.01 if max(means) > 0 else 0.01), f"{val:.3f}", ha="center", va="bottom", fontsize=9)
            fname = os.path.join(out_dir, f"{m}_by_distribution.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")

    # 保存 JSON
    try:
        import json
        agg_safe = convert_all(aggregated)
        with open(os.path.join(out_dir, "aggregated_metrics.json"), "w") as f:
            json.dump(agg_safe, f, indent=2)
        print(f"Saved aggregated metrics JSON -> {os.path.join(out_dir,'aggregated_metrics.json')}")
    except Exception as e:
        print("Failed to save aggregated metrics JSON:", e)

    return aggregated

# =========================================================
# CLI + main (保留新版本)
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="DVRP runner with metrics")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gmodel", action="store_true")
    parser.add_argument("--planner", choices=["greedy", "model", "static", "dynamic", "fri", "rbso", "dcp"], default="greedy",
                        help="Planner type: greedy (rule-based), model/static/dynamic (V2Planner), fri, rbso, dcp")
    parser.add_argument("--eval-distributions", action="store_true")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--service-time", action="store_true")
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--map-wid", type=int, default=None)
    parser.add_argument("--map-hei", type=int, default=None)
    parser.add_argument("--total-demand", type=int, default=None)
    parser.add_argument("--static-demands", action="store_true")
    parser.add_argument("--static-ckpt", type=str, default=None, help="Override path to V2 static model checkpoint")
    parser.add_argument("--adapter-ckpt", type=str, default=None, help="Override path to V2 dynamic adapter checkpoint")
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.include_service_time = bool(args.service_time)
    if args.num_agents is not None and args.num_agents > 0:
        cfg.num_agents = int(args.num_agents)
    if args.map_wid is not None and args.map_wid > 0:
        cfg.width = int(args.map_wid)
    if args.map_hei is not None and args.map_hei > 0:
        cfg.height = int(args.map_hei)
    if args.total_demand is not None and args.total_demand > 0:
        cfg.generator_params["total_demand"] = int(args.total_demand)

    # Simplified planner selection for V2Planner architecture
    planner_choice = args.planner
    
    # Map "model" to "dynamic" for backwards compatibility
    if planner_choice == "model":
        planner_choice = "dynamic"
    
    cfg.planner_type = planner_choice
    
    # Handle V2Planner checkpoint overrides
    if planner_choice in ("static", "dynamic"):
        if not hasattr(cfg, 'v2_planner_params'):
            cfg.v2_planner_params = {}
        if args.static_ckpt:
            cfg.v2_planner_params["static_ckpt"] = args.static_ckpt
        if args.adapter_ckpt and planner_choice == "dynamic":
            cfg.v2_planner_params["adapter_ckpt"] = args.adapter_ckpt

    if args.gmodel:
        cfg.generator_type = "net"

    if args.eval_distributions:
        print("Starting distribution evaluation...")
        evaluate_distributions(cfg, planner_choice, num_runs=args.num_runs, out_dir="outputs/eval")
        return

    metrics = run_episode_return_metrics(cfg, seed=args.seed, render=args.render, fps=args.fps, planner=planner_choice, static_demands=args.static_demands)
    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        if k != "paths":
            print(f"{k:25s}: {v}")

    import json
    safe_paths = convert_all(metrics["paths"])
    # Save per-run outputs under `outputs/run/<timestamp>/`
    output_root = os.path.join("outputs", "run")
    run_run_dir = create_output_run_dir(output_root, "run")
    # Save full metrics JSON
    try:
        metrics_safe = convert_all(metrics)
        with open(os.path.join(run_run_dir, "metrics.json"), "w") as f:
            json.dump(metrics_safe, f, indent=2)
        with open(os.path.join(run_run_dir, "paths_output.json"), "w") as f:
            json.dump(safe_paths, f, indent=2)
        print(f"\nRun outputs saved -> {run_run_dir}")
    except Exception as e:
        print("Failed to save run outputs:", e)

    # Create plots: (a) agent capacity/load over time, (b) agent cumulative distance over time,
    # (c) served demand (cumulative) over time
    try:
        # (a) loads per step
        loads = metrics.get("loads_per_step", [])
        if loads:
            max_len = max(len(l) for l in loads)
            plt.figure(figsize=(10, 6))
            for i, l in enumerate(loads):
                if len(l) < max_len and len(l) > 0:
                    # extend with last known value
                    l_ext = list(l) + [l[-1]] * (max_len - len(l))
                else:
                    l_ext = list(l)
                plt.plot(range(len(l_ext)), l_ext, label=f"agent_{i}")
            plt.xlabel("time_step")
            plt.ylabel("carried_load")
            plt.title("Agent carried load over time")
            plt.legend(loc="upper right")
            fname = os.path.join(run_run_dir, "loads_over_time.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")

        # (b) cumulative agent distance over time
        paths = metrics.get("paths", [])
        if paths:
            plt.figure(figsize=(10, 6))
            for i, p in enumerate(paths):
                # p is list of (time, x, y, load)
                if not p:
                    continue
                times = [int(t[0]) for t in p]
                xs = [int(t[1]) for t in p]
                ys = [int(t[2]) for t in p]
                cum = []
                acc = 0
                lastx, lasty = xs[0], ys[0]
                cum.append(acc)
                for x, y in zip(xs[1:], ys[1:]):
                    d = abs(int(x) - int(lastx)) + abs(int(y) - int(lasty))
                    acc += d
                    cum.append(acc)
                    lastx, lasty = x, y
                plt.plot(times, cum, label=f"agent_{i}")
            plt.xlabel("time_step")
            plt.ylabel("cumulative_distance")
            plt.title("Agent cumulative movement distance over time")
            plt.legend(loc="upper left")
            fname = os.path.join(run_run_dir, "agent_distance_over_time.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")

        # (c) served demand over time (cumulative)
        served_times = metrics.get("served_times", [])
        # derive timeline length from paths
        max_t = 0
        if paths:
            for p in paths:
                if p:
                    max_t = max(max_t, int(p[-1][0]))
        if served_times and max_t > 0:
            import numpy as _np

            bins = list(range(0, max_t + 2))
            hist, _ = _np.histogram(served_times, bins=bins)
            cum = _np.cumsum(hist)
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(cum)), cum, marker="o")
            plt.xlabel("time_step")
            plt.ylabel("served_requests_cumulative")
            plt.title("Served demand (cumulative) over time")
            fname = os.path.join(run_run_dir, "served_over_time.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")
    except Exception as e:
        print("Failed to generate plots:", e)

if __name__ == "__main__":
    main()
