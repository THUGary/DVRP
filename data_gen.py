from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Dict, Any
from collections import deque

import torch

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.generator import RuleBasedGenerator
from agent.controller import RuleBasedController
from agent.planner import (
    RuleBasedPlanner,
    FastReactiveInserter,
    RepairBasedStabilityOptimizer,
    DistributedCooperativePlanner,
)
from agent.planner.base import AgentState, Target
from utils.state_manager import PlanningState, update_planning_state


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_planner(planner_type: str):
    if planner_type == "greedy":
        return RuleBasedPlanner()
    if planner_type == "fri":
        return FastReactiveInserter()
    if planner_type == "rbso":
        return RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
    if planner_type == "dcp":
        return DistributedCooperativePlanner(auction_rounds=5, bid_strategy="time_urgency")
    raise ValueError(f"Unknown planner type: {planner_type}")


def _unique_nodes_by_xy(observations: List[Tuple[int, int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
    """去重：按 (x,y) 保留第一条"""
    seen = set()
    uniq: List[Tuple[int, int, int, int, int]] = []
    for (x, y, t_arrival, c, t_due) in observations:
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((x, y, t_arrival, c, t_due))
    return uniq


def _targets_to_next_label(
    next_target: Tuple[int, int] | None,
    nodes_unique: List[Tuple[int, int, int, int, int]],
) -> int:
    """将下一目标坐标映射为分类标签索引。未匹配则返回 N 表示 depot。"""
    N = len(nodes_unique)
    if next_target is None:
        return N
    tx, ty = next_target
    for i, (x, y, *_rest) in enumerate(nodes_unique):
        if (x, y) == (tx, ty):
            return i
    return N  # 不在节点中，视为 depot


def collect_rows_from_call(
    time_now: int,
    observations: List[Tuple[int, int, int, int, int]],
    agent_states_xyz: List[Tuple[int, int, int]],
    depot_xy: Tuple[int, int],
    horizon: int,
    current_plans: List[deque[Tuple[int, int]]],
    global_nodes: List[Tuple[int, int, int, int, int]] | None,
    serve_mark: List[int] | None,
    unserved_count: int | None,
    targets: List[deque[Tuple[int, int]]],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """从一次 planner 调用组装每个 agent 的监督样本 row"""
    nodes_unique = _unique_nodes_by_xy(observations)
    N = len(nodes_unique)
    rows: List[Dict[str, Any]] = []

    for agent_id, (ax, ay, s) in enumerate(agent_states_xyz):
        # 下一目标：队首，若空 None
        next_xy = targets[agent_id][0] if len(targets[agent_id]) > 0 else None
        label = _targets_to_next_label(next_xy, nodes_unique)

        row = {
            "nodes": nodes_unique,                    # List[(x, y, t_arrival, c, t_due)]
            "node_mask": [False] * N,                 # 简化版，全部可选；pad 时会额外mask
            "agent": (ax, ay, s, time_now),           # (x, y, s, t_agent)
            "depot": (depot_xy[0], depot_xy[1], time_now),
            "label": label,                           # 0..N, N==depot
            "valid_N": N,
            "planner_inputs": {
                "time": time_now,
                "horizon": horizon,
                "current_plans": [list(q) for q in current_plans],
                # 注意：global_nodes 在 state_manager 中为 (x,y,t_arrival,t_due,demand)
                "global_nodes": list(global_nodes) if global_nodes is not None else [],
                "serve_mark": list(serve_mark) if serve_mark is not None else None,
                "unserved_count": int(unserved_count) if unserved_count is not None else None,
                "depot": depot_xy,
            },
            "meta": meta | {"agent_id": agent_id},
        }
        rows.append(row)
    return rows


def generate_dataset(
    cfg: Config,
    episodes: int,
    planner_type: str,
    seed: int,
    out_dir: str,
    val_ratio: float,
    replan_policy: str = "always",  # "always" | "on_new_or_empty"
) -> Dict[str, str]:
    """运行 episodes，收集 rows 并落盘"""
    _ensure_dir(out_dir)
    all_rows: List[Dict[str, Any]] = []
    rng = None

    for ep in range(episodes):
        gen = RuleBasedGenerator(cfg.width, cfg.height, **cfg.generator_params)
        env = GridEnvironment(
            width=cfg.width,
            height=cfg.height,
            num_agents=cfg.num_agents,
            capacity=cfg.capacity,
            depot=cfg.depot,
            generator=gen,
            max_time=cfg.max_time,
        )
        env.num_agents = cfg.num_agents
        planner = _build_planner(planner_type)
        controller = RuleBasedController(**cfg.controller_params)

        obs = env.reset(seed + ep)
        planning_state = PlanningState()
        planning_state.reset(cfg.num_agents)

        prev_demands = []
        done = False

        step_id = 0
        while not done:
            demands = obs["demands"]  # [(x, y, t, c, end_t), ...]
            new_demands = [d for d in demands if d not in prev_demands]

            agent_states = obs["agent_states"]  # [(x,y,s), ...]
            update_planning_state(
                planning_state=planning_state,
                agent_states=agent_states,
                new_demands=new_demands,
                obs_demands=demands,
            )

            # 确定是否重规划
            can_continue = all(len(q) > 0 for q in planning_state.current_plans)
            need_replan = True if replan_policy == "always" else (len(new_demands) > 0 or not can_continue)

            # 计划（兼容 BasePlanner.plan 的全部参数）
            if need_replan:
                agents = [AgentState(x=a[0], y=a[1], s=a[2]) for a in agent_states]
                targets = planner.plan(
                    observations=demands,
                    agent_states=agents,
                    depot=obs["depot"],
                    t=obs["time"],
                    horizon=1,
                    current_plans=planning_state.current_plans,
                    global_nodes=planning_state.global_nodes.nodes,
                    serve_mark=planning_state.global_nodes.serve_mark,
                    unserved_count=planning_state.get_unserved_count(),
                )
                planning_state.update_plans(targets)
            else:
                targets = planning_state.current_plans

            # 收集监督样本 rows（以“下一步目标”作为标签）
            meta = {
                "episode_id": ep,
                "step_id": step_id,
                "planner": planner_type,
                "map_wid": cfg.width,
                "map_hei": cfg.height,
                "agent_num": cfg.num_agents,
                "capacity": cfg.capacity,
                "max_time": cfg.max_time,
            }
            rows = collect_rows_from_call(
                time_now=obs["time"],
                observations=demands,
                agent_states_xyz=agent_states,
                depot_xy=obs["depot"],
                horizon=1,
                current_plans=planning_state.current_plans,
                global_nodes=planning_state.global_nodes.nodes,
                serve_mark=planning_state.global_nodes.serve_mark,
                unserved_count=planning_state.get_unserved_count(),
                targets=targets,
                meta=meta,
            )
            all_rows.extend(rows)

            # 环境前进一步
            # 让每个 agent 朝队首目标移动一步（与 train.py 一致）
            actions: List[Tuple[int, int]] = []
            for i, (x, y, s) in enumerate(agent_states):
                # 弹出已到达
                while len(planning_state.current_plans[i]) > 0 and planning_state.current_plans[i][0] == (x, y):
                    planning_state.current_plans[i].popleft()
                if len(planning_state.current_plans[i]) == 0:
                    actions.append((0, 0))
                else:
                    actions.append(controller.act((x, y), planning_state.current_plans[i]))
            obs, reward, done, info = env.step(actions)
            prev_demands = list(demands)
            step_id += 1

    # 拆分 Train/Val
    total = len(all_rows)
    n_val = int(total * val_ratio)
    val_rows = all_rows[:n_val]
    trn_rows = all_rows[n_val:]

    prefix = "plans"
    train_path = os.path.join(out_dir, f"{prefix}_train_{cfg.width}_{cfg.num_agents}.pt")
    val_path = os.path.join(out_dir, f"{prefix}_val_{cfg.width}_{cfg.num_agents}.pt")
    payload = {
        "rows": trn_rows,
        "meta": {"total_rows": len(trn_rows), "val_rows": len(val_rows), "version": "v1"},
    }
    torch.save(payload, train_path)
    torch.save({"rows": val_rows, "meta": payload["meta"]}, val_path)
    print(f"[DATA] saved train={train_path} ({len(trn_rows)}) val={val_path} ({len(val_rows)})")
    return {"train": train_path, "val": val_path}


def main():
    ap = argparse.ArgumentParser(description="Generate supervised rows by running planner in env")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--planner", type=str, default="greedy", choices=["greedy", "fri", "rbso", "dcp"])
    ap.add_argument("--map_wid", type=int, default=20)
    ap.add_argument("--map_hei", type=int, default=20)
    ap.add_argument("--agent_num", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--replan_policy", type=str, default="always", choices=["always", "on_new_or_empty"])
    args = ap.parse_args()

    cfg = get_default_config()
    cfg.width = args.map_wid
    cfg.height = args.map_hei
    cfg.num_agents = args.agent_num

    generate_dataset(
        cfg=cfg,
        episodes=args.episodes,
        planner_type=args.planner,
        seed=args.seed,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        replan_policy=args.replan_policy,
    )


if __name__ == "__main__":
    main()