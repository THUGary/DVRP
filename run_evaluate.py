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
from agent.planner import ModelPlanner
from agent.planner import CVRPPOMOPlanner

import time
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# StaticDemandGenerator (from new code)
# =========================================================
class StaticDemandGenerator:
    """Wrap any BaseDemandGenerator so every demand appears at t=0 (static scenario)."""
    def __init__(self, base_generator):
        self._base = base_generator
        self.width = base_generator.width
        self.height = base_generator.height
        self.params = getattr(base_generator, "params", {})
        self.max_time = getattr(base_generator, "max_time", self.params.get("max_time", 1))
        self._snapshot = []
        self._released = False

    def reset(self, seed: int | None = None):
        self._base.reset(seed)
        self._snapshot.clear()
        self._released = False
        max_time = int(getattr(self._base, "max_time", self.params.get("max_time", 1)))
        max_time = max(1, max_time)
        for t in range(max_time):
            for demand in self._base.sample(t):
                self._snapshot.append(demand)

    def sample(self, t: int):
        if t == 0 and not self._released:
            self._released = True
            return list(self._snapshot)
        return []

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
        self._request_registry: Dict[Tuple[int, int, int], int] = {}
        self.load_history: List[List[int]] = [[] for _ in range(num_agents)]

    @staticmethod
    def _tuple_key_from_d(d):
        if d is None:
            return None
        if isinstance(d, (list, tuple)) and len(d) >= 3:
            try:
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
                    self._request_registry[key] = int(current_time)
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
        key = (int(x), int(y), int(t))
        req_time = self._request_registry.pop(key, None)
        if req_time is not None:
            rt = int(served_time) - int(req_time)
            if rt >= 0:
                self.response_times.append(rt)
        self.served_requests += 1

    def record_expired(self, d_tuple):
        try:
            x, y, t, c, end_t = d_tuple
            trip_key = (int(x), int(y), int(t))
            if trip_key in self._request_registry:
                self._request_registry.pop(trip_key, None)
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

# =========================================================
def build_env(cfg: Config, planner_type: str, *, static_demands: bool = False) -> Tuple[GridEnvironment, BaseDemandGenerator, BasePlanner, RuleBasedController]:
    if cfg.generator_type == "net":
        from agent.generator.net_generator import NetDemandGenerator as GenClass
    else:
        from agent.generator import RuleBasedGenerator as GenClass
    gen = GenClass(cfg.width, cfg.height, **cfg.generator_params)
    if static_demands:
        gen = StaticDemandGenerator(gen)
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
        max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
        include_service_time=bool(getattr(cfg, "include_service_time", False)),
    )
    env.num_agents = cfg.num_agents

    if planner_type == "greedy":
        planner = RuleBasedPlanner(full_capacity=cfg.capacity, **cfg.planner_params)
    elif planner_type == "fri":
        planner = FastReactiveInserter()
    elif planner_type == "rbso":
        planner = RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
    elif planner_type == "dcp":
        planner = DistributedCooperativePlanner(auction_rounds=5, bid_strategy='time_urgency')
    elif planner_type == "model":
        planner_params = dict(cfg.model_planner_params)
        ckpt_path = planner_params.pop("ckpt", None)
        planner = ModelPlanner(full_capacity=cfg.capacity, **planner_params)
        if ckpt_path:
            if hasattr(planner, "load_from_ckpt"):
                planner.load_from_ckpt(ckpt_path)
            else:
                raise RuntimeError("Selected planner does not support checkpoint loading.")
    elif planner_type == "cvrp_pomo":
        cvrp_params = dict(cfg.cvrp_planner_params)
        cvrp_params.pop("enabled", None)
        pomo_root = cvrp_params.pop("pomo_root", None)
        if not pomo_root:
            raise ValueError("Config.cvrp_planner_params must define 'pomo_root' when using the CVRP planner.")
        env_params = copy.deepcopy(cvrp_params.pop("env_params", {}))
        model_params = copy.deepcopy(cvrp_params.pop("model_params", {}))
        checkpoint = cvrp_params.pop("checkpoint", None)
        device_override = cvrp_params.pop("device", "cpu")
        max_nodes = cvrp_params.pop("max_nodes", env_params.get("problem_size", cfg.capacity))
        coord_norm = cvrp_params.pop("coord_normalizer", None)
        selection_policy = cvrp_params.pop("selection_policy", "earliest_due")
        planner = CVRPPOMOPlanner(
            pomo_root=pomo_root,
            env_params=env_params,
            model_params=model_params,
            checkpoint=checkpoint,
            device=device_override,
            max_nodes=max_nodes,
            coord_normalizer=coord_norm,
            grid_width=cfg.width,
            grid_height=cfg.height,
            capacity=cfg.capacity,
            selection_policy=selection_policy,
            **cvrp_params,
        )
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
    controller = RuleBasedController(**cfg.controller_params)
    return env, gen, planner, controller

# =========================================================
# run_episode + EvaluationTracker metrics (融合新旧)
# =========================================================
def run_episode_return_metrics(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, planner: str = "greedy", *, static_demands: bool = False):
    rng = random.Random(seed)
    depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))
    cfg = copy.deepcopy(cfg)
    cfg.depot = depot
    cfg.generator_params = {**cfg.generator_params, "depot": depot}
    planner_type = planner
    env, gen, planner_impl, controller = build_env(cfg, planner_type, static_demands=static_demands)
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

    if render:
        renderer = PygameRenderer(cfg.width, cfg.height)
        renderer.init()

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
        )

        agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
        plan_horizon = 1 if planner_type != "model" else max(1, int(cfg.model_planner_params.get("time_plan", 1)))
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

    if renderer:
        renderer.close()

    metrics = tracker.finalize()
    return metrics

# =========================================================
# evaluate_distributions (迁移旧 run.py)
# =========================================================
def evaluate_distributions(cfg: Config, planner_choice: str, num_runs: int = 10, out_dir: str = "outputs/eval"):
    os.makedirs(out_dir, exist_ok=True)
    distributions = ["uniform", "gaussian", "cluster", "explosion", "implosion"]
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
    parser.add_argument("--pmodel", nargs="?", const="__DEFAULT__")
    parser.add_argument("--gmodel", action="store_true")
    parser.add_argument("--planner", choices=["greedy", "model", "cvrp_pomo", "fri", "rbso", "dcp"], default=None)
    parser.add_argument("--eval-distributions", action="store_true")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--service-time", action="store_true")
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--map-wid", type=int, default=None)
    parser.add_argument("--map-hei", type=int, default=None)
    parser.add_argument("--total-demand", type=int, default=None)
    parser.add_argument("--cvrp", action="store_true")
    parser.add_argument("--cvrp-root", type=str, default=None)
    parser.add_argument("--cvrp-ckpt", type=str, default=None)
    parser.add_argument("--static-demands", action="store_true")
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

    # Planner选择逻辑
    planner_choice = None
    use_model_planner = False
    if args.planner is not None:
        planner_choice = args.planner
        if planner_choice == "model":
            use_model_planner = True
            cfg.planner_type = "model"
        elif planner_choice == "cvrp_pomo":
            cfg.planner_type = "cvrp_pomo"
        else:
            cfg.planner_type = planner_choice
    else:
        use_model_planner = args.pmodel is not None
        if args.cvrp and use_model_planner:
            raise ValueError("--cvrp and --pmodel are mutually exclusive")
        if args.cvrp:
            planner_choice = "cvrp_pomo"
            cfg.planner_type = "cvrp_pomo"
            if args.cvrp_root:
                cfg.cvrp_planner_params["pomo_root"] = args.cvrp_root
            if args.cvrp_ckpt:
                cfg.cvrp_planner_params["checkpoint"] = args.cvrp_ckpt
        else:
            planner_choice = "model" if use_model_planner else "greedy"
            if use_model_planner:
                cfg.planner_type = "model"

    # model planner ckpt解析
    if use_model_planner and isinstance(args.pmodel, str) and args.pmodel != "__DEFAULT__":
        raw = os.path.expanduser(args.pmodel)
        candidates = [
            raw,
            os.path.join("checkpoints", raw) if not os.path.isabs(raw) else raw,
            os.path.join("checkpoints", "planner", os.path.basename(raw)),
        ]
        ckpt_path = None
        for p in candidates:
            p_abs = os.path.abspath(p)
            if os.path.isfile(p_abs):
                ckpt_path = p_abs
                break
        if ckpt_path is None:
            ckpt_path = os.path.abspath(raw)
            print(f"WARNING: Planner checkpoint not found at {ckpt_path}. Will attempt to load; ensure path is correct.")
        cfg.model_planner_params["ckpt"] = ckpt_path

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
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/paths_output.json", "w") as f:
        json.dump(safe_paths, f, indent=2)
    print("\nPaths saved → outputs/paths_output.json")

if __name__ == "__main__":
    main()
