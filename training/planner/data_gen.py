from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Dict, Any
from collections import deque

import torch
import math

# Ensure project root on sys.path from nested training directory
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import numpy as _np  # 仅用于保存前的类型规整；加载时无需 numpy
except Exception:  # numpy 可选依赖
    _np = None

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


def _to_builtin_number(x: Any) -> Any:
    """将 numpy 标量/torch 标量 转换为内置 Python number，其他类型原样返回。"""
    # numpy 标量
    if _np is not None and isinstance(x, _np.generic):
        return x.item()
    # torch 标量张量
    if isinstance(x, torch.Tensor) and x.dim() == 0:
        return x.item()
    return x


def _sanitize_for_torch_save(obj: Any) -> Any:
    """
    递归地将对象转换为 PyTorch 2.6+ 在默认 weights_only 加载下也安全/可兼容的结构：
    - 仅包含基础类型(int/float/bool/str/None)、
      以及由 list/tuple/dict 组合的容器；
    - 对 numpy.ndarray 转换为嵌套 Python 列表（其中元素再递归规整为内置标量）；
    - 对 numpy 标量/torch 0D 张量转换为内置标量；
    - 对 torch 张量保留为张量（weights_only 允许张量）；
    - 对 deque/set 转换为 list。
    注意：不引入任何用户自定义类实例，避免触发反序列化限制。
    """
    # None 或基础类型
    if obj is None or isinstance(obj, (bool, int, float, str)):
        # 防止 NaN/Inf 作为 float 落盘造成解析差异：统一为 Python float
        if isinstance(obj, float):
            if math.isnan(obj):
                return float("nan")
            if math.isinf(obj):
                return float("inf") if obj > 0 else float("-inf")
        return obj

    # numpy 数组
    if _np is not None and isinstance(obj, _np.ndarray):
        # 转换为嵌套 list，并递归规整元素
        return _sanitize_for_torch_save(obj.tolist())

    # numpy 标量 / torch 标量
    builtin_num = _to_builtin_number(obj)
    if builtin_num is not obj:
        return builtin_num

    # torch 张量（非标量）
    if isinstance(obj, torch.Tensor):
        return obj.contiguous()

    # 容器类型
    if isinstance(obj, (list, tuple, set, deque)):
        seq = list(obj)  # 统一为 list
        return [_sanitize_for_torch_save(v) for v in seq]

    if isinstance(obj, dict):
        # 键尽量转为 str，值递归规整
        new_dict: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, (int, float, bool)):
                k = str(k)
            elif not isinstance(k, str):
                k = str(k)
            new_dict[k] = _sanitize_for_torch_save(v)
        return new_dict

    # 元组(如坐标)可能出现在上面的分支；若落到这里，兜底转字符串避免自定义类型
    return str(obj)


def _build_planner(planner_type: str, capacity: int | None = None):
    if planner_type == "greedy":
        return RuleBasedPlanner(full_capacity=capacity)
    if planner_type == "fri":
        return FastReactiveInserter()
    if planner_type == "rbso":
        return RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
    if planner_type == "dcp":
        return DistributedCooperativePlanner(auction_rounds=5, bid_strategy="time_urgency")
    raise ValueError(f"Unknown planner type: {planner_type}")


def _unique_nodes_by_xy(observations: List[Tuple[int, int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
    """去重：按 (x,y) 保留第一条，且忽略 c<=0 的点（与 RuleBasedPlanner 的可行性过滤一致）。"""
    seen = set()
    uniq: List[Tuple[int, int, int, int, int]] = []
    for (x, y, t_arrival, c, t_due) in observations:
        if c <= 0:
            continue
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((x, y, t_arrival, c, t_due))
    return uniq


def _target_to_label(
    target: Tuple[int, int] | None,
    nodes_unique: List[Tuple[int, int, int, int, int]],
) -> int:
    """将目标坐标映射为分类标签索引。约定 depot=0，nodes 为 1..N（对应 nodes_unique[0..N-1]）。"""
    N = len(nodes_unique)
    if target is None:
        return 0
    tx, ty = target
    for i, (x, y, *_rest) in enumerate(nodes_unique):
        if (x, y) == (tx, ty):
            return i + 1
    return 0

def _targets_to_k_labels(
    tgt_deque: deque[Tuple[int, int]],
    nodes_unique: List[Tuple[int, int, int, int, int]],
    k: int,
) -> List[int]:
    labels: List[int] = []
    for step in range(k):
        xy = tgt_deque[step] if step < len(tgt_deque) else None
        labels.append(_target_to_label(xy, nodes_unique))
    return labels


def collect_rows_from_call(
    time_now: int,
    observations: List[Tuple[int, int, int, int, int]],
    agent_states_xyz: List[Tuple[int, int, int]],
    depot_xy: Tuple[int, int],
    k: int,
    current_plans: List[deque[Tuple[int, int]]],
    global_nodes: List[Tuple[int, int, int, int, int]] | None,
    serve_mark: List[int] | None,
    unserved_count: int | None,
    targets: List[deque[Tuple[int, int]]],
    meta: Dict[str, Any],
    full_capacity: int,
) -> Dict[str, Any]:
    """从一次 planner 调用组装一个包含所有 agents 的监督样本 row：labels 形状为 [A, K]。"""
    nodes_unique = _unique_nodes_by_xy(observations)
    # 校验：禁止需求点与 depot 坐标重合（硬错误，避免错误样本进入数据集）
    for (x, y, _ta, _c, _due) in nodes_unique:
        if (x, y) == tuple(depot_xy):
            raise ValueError(
                f"[DATA-CHECK][depot-overlap] Encountered a demand at depot position {depot_xy} at time={time_now}. "
                f"Please adjust generator or map size to avoid overlaps."
            )
    N = len(nodes_unique)
    A = len(agent_states_xyz)

    # agents 列表与二位标签 [A, K]
    agents_list: List[Tuple[int, int, int, int]] = []
    labels_ak: List[List[int]] = []
    for agent_id, (ax, ay, s) in enumerate(agent_states_xyz):
        agents_list.append((ax, ay, s, time_now))
        labels_k = _targets_to_k_labels(targets[agent_id], nodes_unique, k)
        # debug
        # print(f"Agent {agent_id} targets: {list(targets[agent_id])} -> labels: {labels_k}")
        labels_ak.append(labels_k)

    # 基本行数据
    row: Dict[str, Any] = {
        "nodes": nodes_unique,                    # List[(x, y, t_arrival, c, t_due)]
        "node_mask": [False] * N,                 # 简化版，全部可选；pad 时会额外mask
        "agents": agents_list,                    # List[(x, y, s, t_agent)], 长度 A
        "depot": (depot_xy[0], depot_xy[1], time_now),
        "labels_ak": labels_ak,                   # List[List[int]], 形状 [A, K]，N==depot
        "full_capacity": int(full_capacity),      # 每个 agent 回 depot 恢复的满容量
        "valid_N": N,
        "planner_inputs": {
            "time": time_now,
            "k": k,
            "current_plans": [list(q) for q in current_plans],
            # 注意：global_nodes 在 state_manager 中为 (x,y,t_arrival,t_due,demand)
            "global_nodes": list(global_nodes) if global_nodes is not None else [],
            "serve_mark": list(serve_mark) if serve_mark is not None else None,
            "unserved_count": int(unserved_count) if unserved_count is not None else None,
            "depot": depot_xy,
        },
    "meta": {**meta, "agent_num": A},
    }
    # 调试输出（可按需开启）
    # print(f"[DATA-COLLECT] ep={meta.get('episode_id')} step={meta.get('step_id')}")
    # print(f"agents: {agents_list}")
    # print(f"depot: {depot_xy} {time_now}")
    # print(f"labels_ak: {labels_ak}")
    # print(f"nodes_unique: {nodes_unique}")

    # 额外一致性校验（仅日志，不修改标签）：确保 labels_ak 不违反容量约束
    # 规则：从各自 agent 的当前空间 s 出发，按 labels_ak 顺序执行；若命中 depot(0)，则将空间恢复为 full_capacity；
    # 若命中某节点 i (1<=i<=N)，则需要 nodes_unique[i-1][3] <= 当前空间。
    try:
        demands_vec = [int(x[3]) for x in nodes_unique]  # c 字段
        for a_idx, (ax, ay, s, _ta) in enumerate(agents_list):
            space = int(s)
            trace_space_before: List[int] = []
            trace_space_after: List[int] = []
            trace_labels: List[int] = []
            trace_demands: List[int] = []
            violated_detail = None
            for step_idx, lab in enumerate(labels_ak[a_idx]):
                trace_space_before.append(space)
                trace_labels.append(lab)
                if lab == 0:
                    trace_demands.append(-1)
                    space = int(full_capacity)
                    trace_space_after.append(space)
                    continue
                if 1 <= lab <= N:
                    req = demands_vec[lab - 1]
                    trace_demands.append(int(req))
                    if req > space and violated_detail is None:
                        violated_detail = (step_idx, lab, int(req), int(space))
                    space = max(0, space - req)
                    trace_space_after.append(space)
                else:
                    trace_demands.append(-2)
                    space = int(full_capacity)
                    trace_space_after.append(space)
            if violated_detail is not None:
                v_step, v_lab, v_req, v_space = violated_detail
                # 输出更详细的轨迹，便于定位原因
                print(
                    f"[DATA-CHECK][cap-violation] ep={meta.get('episode_id')} step={meta.get('step_id')} agent={a_idx} "
                    f"k={v_step} label_node={v_lab} demand={v_req} space={v_space} — labels imply demand>space\n"
                    f"  labels_k={trace_labels}\n  demands@labels={trace_demands}\n  space_before={trace_space_before}\n  space_after={trace_space_after}"
                )
                # 返回错误
                raise ValueError(
                    f"Capacity violation detected for agent {a_idx} at step {v_step} in episode {meta.get('episode_id')} step {meta.get('step_id')}: "
                    f"label_node={v_lab}, demand={v_req}, available_space={v_space}."
                )
    except Exception as e:
        print(f"[DATA-CHECK] capacity validation skipped due to error: {e}")
    return row


def generate_dataset(
    cfg: Config,
    episodes: int,
    planner_type: str,
    seed: int,
    out_dir: str,
    val_ratio: float,
    replan_policy: str = "always",  # "always" | "on_new_or_empty"
    k: int = 3,
) -> Dict[str, str]:
    """运行 episodes，收集 rows 并落盘"""
    _ensure_dir(out_dir)
    all_rows: List[Dict[str, Any]] = []
    rng = None

    for ep in range(episodes):
        # 将 depot 传入生成器，避免生成与 depot 重叠的需求
        # cfg.generator_params 可能已包含 'depot'（在 configs.__post_init__ 中替换占位符），
        # 为避免重复关键字参数导致 TypeError，我们先复制并弹出可能的重复项。
        gen_params = dict(cfg.generator_params) if cfg.generator_params is not None else {}
        gen_params.pop("depot", None)
        gen = RuleBasedGenerator(cfg.width, cfg.height, depot=cfg.depot, **gen_params)
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
        planner = _build_planner(planner_type, capacity=cfg.capacity)
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
                    horizon=k,
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
            row = collect_rows_from_call(
                time_now=obs["time"],
                observations=demands,
                agent_states_xyz=agent_states,
                depot_xy=obs["depot"],
                k=k,
                current_plans=planning_state.current_plans,
                global_nodes=planning_state.global_nodes.nodes,
                serve_mark=planning_state.global_nodes.serve_mark,
                unserved_count=planning_state.get_unserved_count(),
                targets=targets,
                meta=meta,
                full_capacity=cfg.capacity,
            )
            all_rows.append(row)

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
    # 保存前进行类型规整，确保不包含 numpy 标量/数组、deque 或自定义对象
    payload = {
        "rows": trn_rows,
        "meta": {
            "total_rows": len(trn_rows),
            "val_rows": len(val_rows),
            "version": "v2",
            "weights_only_compatible": True,
        },
    }
    safe_train = _sanitize_for_torch_save(payload)
    safe_val = _sanitize_for_torch_save({"rows": val_rows, "meta": payload["meta"]})

    torch.save(safe_train, train_path)
    torch.save(safe_val, val_path)
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
    ap.add_argument("--k", type=int, default=3, help="每个监督样本的未来步数标签")
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
        k=args.k,
    )


if __name__ == "__main__":
    main()