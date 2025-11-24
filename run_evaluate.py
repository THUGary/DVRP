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

import time
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
#  EvaluationTracker —— 已修复全部指标问题
# =========================================================
class EvaluationTracker:
    """
    Tracks:
      - total_distance
      - agent_distance
      - avg_route_length
      - route_balance_std
      - vehicles_used
      - total_requests
      - served_requests
      - expired_requests
      - service_ratio
      - avg_response_time
      - paths
    """

    def __init__(self, num_agents: int):
        self.num_agents = num_agents

        # distance
        self.last_positions = [None] * num_agents
        self.total_distance = 0.0
        self.agent_distance = [0.0] * num_agents

        # paths
        self.paths = [[] for _ in range(num_agents)]

        # demand stats
        self.total_requests = 0
        self.served_requests = 0
        self.expired_requests = 0
        self.response_times: List[int] = []

        # demand 注册表，key 为 (x, y, t)
        # value 为 demand 出现时间
        self._request_registry: Dict[Tuple[int, int, int], int] = {}

        # per-agent loads per step/trip
        # 每个 agent 的每一步载量（车上承载的需求数量）
        self.load_history: List[List[int]] = [[] for _ in range(num_agents)]



    # -----------------------------
    # Register new demands
    # -----------------------------
     # 注册新出现的 demand
    def register_new_demands(self, new_demands, current_time: int):
        self.total_requests += 1
        for d in new_demands:
            try:
                x, y, t, c, end_t = d
                key = (int(x), int(y), int(t))
                if key not in self._request_registry:
                    self._request_registry[key] = int(current_time)
                    # 与 served_requests 逻辑一致，每个 demand 第一次出现即计入 total_requests
                
            except Exception:
                continue

    # -----------------------------
    # Record movement
    # -----------------------------
    def record_path_and_distance(self, agent_states, time_step: int, env):
        for i, (x, y, load) in enumerate(agent_states):
            self.paths[i].append((int(time_step), int(x), int(y), int(load)))
            self.load_history[i].append(int(load))
            last = self.last_positions[i]
            if last is not None:
                lx, ly = last
                d = abs(int(x) - int(lx)) + abs(int(y) - int(ly))
                self.total_distance += float(d)
                self.agent_distance[i] += float(d)
            self.last_positions[i] = (int(x), int(y))

    # -----------------------------
    # Record served demand
    # -----------------------------
    def record_served_by_tuple(self, d_tuple, served_time: int):
        x, y, t, c, end_t = d_tuple
        key = (int(x), int(y), int(t))
        req_time = self._request_registry.pop(key, None)
        if req_time is not None:
            rt = served_time - req_time
            if rt >= 0:
                self.response_times.append(rt)
        self.served_requests += 1

    # -----------------------------
    # Record expired demand
    # -----------------------------
    def record_expired(self, d_tuple):
        try:
            x, y, t, c, end_t = d_tuple
            trip_key = (int(x), int(y), int(t))
            if trip_key in self._request_registry:
                self._request_registry.pop(trip_key, None)
            else:
                keys_to_remove = [k for k in list(self._request_registry.keys())
                                  if k[0] == int(x) and k[1] == int(y)]
                for k in keys_to_remove:
                    self._request_registry.pop(k, None)
        except Exception:
            pass
        self.expired_requests += 1

    # -----------------------------
    # Finalize metrics
    # -----------------------------
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
            "loads_per_step": list(self.load_history),  # 新增
        }



#  递归转换 numpy → python 类型（保持不变）
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
#  环境构建（保持原样，无任何修改）
# =========================================================
def build_env(cfg: Config, planner_type: str):
    if cfg.generator_type == "net":
        from agent.generator.net_generator import NetDemandGenerator as GenClass
    else:
        from agent.generator import RuleBasedGenerator as GenClass

    gen = GenClass(cfg.width, cfg.height, **cfg.generator_params)

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
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    controller = RuleBasedController(**cfg.controller_params)
    return env, gen, planner, controller


# =========================================================
#  核心 Episode 运行 —— 已修复 served / expired 判定
# =========================================================
def run_episode_return_metrics(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, planner: str = "greedy"):

    rng = random.Random(seed)

    depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))

    cfg = copy.deepcopy(cfg)
    cfg.depot = depot
    cfg.generator_params = {**cfg.generator_params, "depot": depot}

    planner_type = planner
    env, gen, planner_impl, controller = build_env(cfg, planner_type)
    obs = env.reset(seed)

    total_reward = 0.0
    done = False
    step = 0

    renderer = None

    planning_state = PlanningState()
    planning_state.reset(cfg.num_agents)

    # ★ 修正后的指标追踪器
    tracker = EvaluationTracker(cfg.num_agents)

    prev_demands = []

    if render:
        renderer = PygameRenderer(cfg.width, cfg.height)
        renderer.init()

    # ======================================================
    #                 Simulation step loop
    # ======================================================
    while not done:

        current_demands = obs["demands"]

        # ------------------------------------------------------
        # 新需求检测：用于 total_requests + response_time 起点
        # ------------------------------------------------------
        new_demands = [d for d in current_demands if d not in prev_demands]
        

        # ------------------------------------------------------
        # 更新规划状态（保持原样）
        # ------------------------------------------------------
        agent_states = obs["agent_states"]
        update_planning_state(
            planning_state=planning_state,
            agent_states=agent_states,
            new_demands=new_demands,
            obs_demands=current_demands,
        )

        agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
        plan_horizon = 1 if planner_type != "model" else max(1, int(cfg.model_planner_params.get("time_plan", 1)))

        t0 = time.perf_counter()
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
        t1 = time.perf_counter()

        planning_state.update_plans(targets)

        # ------------------
        # 环境 step
        # ------------------
        actions = []
        for i, (x, y, s) in enumerate(agent_states):
            actions.append(controller.act((x, y), targets[i]))

        obs_after, reward, done, info = env.step(actions)

        current_time = obs_after.get("time", 0)
        # =====================================================
        #      判定 disappeared demand 是 served 还是 expired
        # =====================================================
        disappeared = [d for d in prev_demands if d not in obs_after["demands"]]

        for d in disappeared:
            # tuple format: (x, y, t, c, end_t)
            x, y, t, c, end_t = d
            tracker.register_new_demands(d, current_time)
            # 注意：current_time 已经是 env.step 之后的时间
            # 如果 end_t >= current_time -> 说明在到期前被服务（agent 拿走）
            if int(end_t) >= int(current_time):
                # served：以完整 tuple 调用，record_served_by_tuple 会用 (x,y,t) 去匹配注册表
                tracker.record_served_by_tuple(d, served_time=current_time)
            else:
                # expired：调用 record_expired 并让它清理注册表内相关键
                tracker.record_expired(d)

        # =====================================================
        #    记录轨迹 + 距离（保持不变）
        # =====================================================
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
#   批量评估不同分布（保持原样）
# =========================================================
def evaluate_distributions(cfg: Config, planner_choice: str, num_runs: int = 10, out_dir: str = "outputs/eval"):
    os.makedirs(out_dir, exist_ok=True)
    distributions = ["uniform", "gaussian", "cluster", "explosion", "implosion"]

    metric_names = [
        "total_distance",
        "avg_response_time",
        "service_ratio",
        "avg_route_length",
        "vehicles_used"
    ]

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

    # --- 图像保持不动（你暂时不想修改）
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
            plt.text(xi, val + (max(means) * 0.01 if max(means) > 0 else 0.01),
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        fname = os.path.join(out_dir, f"{m}_by_distribution.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot: {fname}")

    # Save JSON
    try:
        import json
        agg_safe = convert_all(aggregated)
        with open(os.path.join(out_dir, "aggregated_metrics.json"), "w") as f:
            json.dump(agg_safe, f, indent=2)
        print(f"Saved aggregated metrics JSON -> {os.path.join(out_dir, 'aggregated_metrics.json')}")
    except Exception as e:
        print("Failed to save aggregated metrics JSON:", e)

    return aggregated


# =========================================================
#   CLI / Main（保持原样）
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="DVRP runner (modified for multi-dist evaluation)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--pmodel", nargs="?", const="__DEFAULT__")
    parser.add_argument("--gmodel", action="store_true")
    parser.add_argument("--eval-distributions", action="store_true")
    parser.add_argument("--num-runs", type=int, default=10)
    args = parser.parse_args()

    cfg = get_default_config()

    use_model_planner = args.pmodel is not None
    planner_choice = "model" if use_model_planner else "greedy"

    if use_model_planner:
        cfg.planner_type = "model"
        if isinstance(args.pmodel, str) and args.pmodel != "__DEFAULT__":
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
                print(f"WARNING: Planner checkpoint not found at {ckpt_path}. Proceeding.")
            cfg.model_planner_params["ckpt"] = ckpt_path

    if args.gmodel:
        cfg.generator_type = "net"

    if args.eval_distributions:
        print("Starting distribution evaluation...")
        evaluate_distributions(cfg, planner_choice, num_runs=args.num_runs, out_dir="outputs/eval")
        return

    metrics = run_episode_return_metrics(cfg, seed=args.seed, render=args.render, fps=args.fps, planner=planner_choice)

    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        if k != "paths":
            print(f"{k:25s}: {v}")

    # Save paths
    import json
    safe_paths = convert_all(metrics["paths"])
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/paths_output.json", "w") as f:
        json.dump(safe_paths, f, indent=2)
    print("\nPaths saved → outputs/paths_output.json")


if __name__ == "__main__":
    main()
