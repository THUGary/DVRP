from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Dict
from collections import deque
from .base import BasePlanner, AgentState, Target
from agent.controller.distance import travel_time


class FastReactiveInserter(BasePlanner):
    """Fast Reactive Inserter (FRI) - 快速响应插入型规划器

    当没有 current_plans 时，使用 Clarke-Wright Savings 算法生成初始解（init_plan）。
    为了避免严格时间窗导致初始化失败，时间窗被转为软约束：
      - 在 savings 计算中引入时间惩罚项（penalty = k * (lateness_after - lateness_before)），
        adjusted_saving = base_saving - penalty
      - 在分配 route 到 agent 时，如果没有完全满足时间窗的 agent，
        会选择使总超时量（lateness sum）最小的 agent（即允许软违约但尽量最小化迟到）

    参数（可通过 planner params 传入）:
      - time_penalty_k: float, 时间惩罚系数 k（默认 1.0）
      - assign_lateness_weight: float, 在 agent 分配选择中对 lateness 的权重（默认 1000.0，
          用来优先考虑减少迟到，作为主度量；可用于在无法完全满足时权衡距离与迟到）

    节点数据结构: (x, y, t_arrival, demand, t_due)
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.time_penalty_k: float = float(params.get("time_penalty_k", 1.0))
        # 在分配 route 到 agent 时，lateness_sum 会被乘以这个权重以形成度量
        # 较大的权重意味着更强的优先减少迟到
        self.assign_lateness_weight: float = float(params.get("assign_lateness_weight", 1000.0))

    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],  # [(x,y,t_arrival,demand,t_due), ...]
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        # Build obs_map keyed by coordinate: store earliest due + demand
        obs_map: Dict[Tuple[int, int], Dict[str, int]] = {}
        for (x, y, t_arrival, demand, t_due) in observations:
            key = (x, y)
            prev = obs_map.get(key)
            # keep the one with the earliest due time (more urgent)
            if prev is None or t_due < prev["t_due"]:
                obs_map[key] = {"t_arrival": t_arrival, "demand": demand, "t_due": t_due}

        num_agents = len(agent_states)

        # If there is no meaningful current plan, build an initial high-quality plan via Clarke-Wright
        if current_plans is None or all((p is None or len(p) == 0) for p in current_plans):
            return self.init_plan(obs_map, agent_states, depot, t)

        # Otherwise, do reactive insertion on top of current_plans
        current_plans = [deque(p) for p in current_plans]

        # assigned nodes set (by coord)
        assigned_nodes = set()
        for p in current_plans:
            for n in p:
                assigned_nodes.add(n)

        # collect unassigned active demands
        unassigned = []
        for coord, info in obs_map.items():
            if info["demand"] > 0 and coord not in assigned_nodes:
                unassigned.append((coord[0], coord[1], info["demand"], info["t_arrival"], info["t_due"]))

        # helper to compute ETAs for a list of route points starting from start_pos at start_time
        def compute_etas(start_pos: Tuple[int, int], route_points: List[Tuple[int, int]], start_time: int) -> List[int]:
            etas: List[int] = []
            cur = start_pos
            cur_time = start_time
            for rp in route_points:
                cur_time += travel_time(cur, rp)
                etas.append(cur_time)
                cur = rp
            return etas

        # Greedy insertion of each unassigned node: find best vehicle & insertion pos with time-window *soft* consideration
        for (nx, ny, demand, t_arrival, t_due) in unassigned:
            best = {"score": float("inf"), "vehicle": -1, "pos": -1}
            for vidx, agent in enumerate(agent_states):
                # capacity check (strict)
                if agent.s < demand:
                    continue
                # vehicle current position and plan
                plan = list(current_plans[vidx])
                route = [(agent.x, agent.y)] + plan
                # try insertion positions (insert between route[i-1] and route[i], plan index = pos-1)
                for insert_pos in range(1, len(route) + 1):
                    prev = route[insert_pos - 1]
                    if insert_pos < len(route):
                        nxt = route[insert_pos]
                        original = travel_time(prev, nxt)
                        newd = travel_time(prev, (nx, ny)) + travel_time((nx, ny), nxt)
                        delta = newd - original
                    else:
                        last = route[-1]
                        delta = travel_time(last, (nx, ny))
                    # build new plan points (excluding vehicle pos)
                    new_plan_points = plan.copy()
                    plan_insert_idx = insert_pos - 1
                    new_plan_points.insert(plan_insert_idx, (nx, ny))
                    # compute lateness sum as soft measure (sum of max(0, ETA - t_due))
                    etas = compute_etas((agent.x, agent.y), new_plan_points, t)
                    lateness_sum = 0.0
                    for rp, eta in zip(new_plan_points, etas):
                        info = obs_map.get(rp)
                        if info:
                            lateness = max(0, eta - info["t_due"])
                            lateness_sum += lateness
                    # score: primary is delta (extra travel), secondary is lateness weighted
                    score = delta + (self.assign_lateness_weight * lateness_sum)
                    if score < best["score"]:
                        best = {"score": score, "vehicle": vidx, "pos": plan_insert_idx}
            # apply best insertion if found
            if best["vehicle"] >= 0:
                vplan = list(current_plans[best["vehicle"]])
                if best["pos"] >= len(vplan):
                    vplan.append((nx, ny))
                else:
                    vplan.insert(best["pos"], (nx, ny))
                current_plans[best["vehicle"]] = deque(vplan)

        # ensure every vehicle has at least a depot target
        for idx, p in enumerate(current_plans):
            if len(p) == 0:
                p.append(depot)

        return current_plans

    def init_plan(
        self,
        obs_map: Dict[Tuple[int, int], Dict[str, int]],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        current_time: int,
    ) -> List[Deque[Target]]:
        """
        Build initial solution using Clarke-Wright Savings algorithm with soft time-window penalty.

        obs_map: {(x,y): {"t_arrival":..., "demand":..., "t_due":...}, ...}
        Returns a list[deque[(x,y)]] of length = num_agents, assigned to agent indices.
        """
        # Helper functions
        def compute_etas(start_pos: Tuple[int, int], route_points: List[Tuple[int, int]], start_time: int) -> List[int]:
            etas = []
            cur = start_pos
            cur_time = start_time
            for p in route_points:
                cur_time += travel_time(cur, p)
                etas.append(cur_time)
                cur = p
            return etas

        def lateness_for_sequence(start_pos: Tuple[int, int], seq: List[Tuple[int, int]], start_time: int) -> float:
            """Sum of positive lateness (ETA - t_due) for a sequence starting from start_pos at start_time."""
            etas = compute_etas(start_pos, seq, start_time)
            s = 0.0
            for node, eta in zip(seq, etas):
                info = obs_map.get(node)
                if info:
                    s += max(0.0, eta - info["t_due"])
            return s

        def route_demand(route: List[Tuple[int, int]]) -> int:
            s = 0
            for n in route:
                s += obs_map.get(n, {}).get("demand", 0)
            return s

        customers = list(obs_map.keys())
        num_agents = len(agent_states)
        if not customers:
            # no customers => return depot-only plans
            return [deque([depot]) for _ in agent_states]

        # maximum capacity to consider for merging (use max agent capacity to allow merges)
        max_cap = max(a.s for a in agent_states) if agent_states else 0

        # initial routes: each customer in its own route
        routes: List[List[Tuple[int, int]]] = [[c] for c in customers]
        # mapping from customer -> route index
        cust_to_route: Dict[Tuple[int, int], int] = {c: i for i, c in enumerate(customers)}

        # compute savings with time-penalty adjustment
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                base_sij = travel_time(depot, ci) + travel_time(depot, cj) - travel_time(ci, cj)
                # lateness if served separately from depot
                lateness_sep = lateness_for_sequence(depot, [ci], current_time) + lateness_for_sequence(depot, [cj], current_time)
                # lateness if merged (ci then cj) from depot
                lateness_merged = lateness_for_sequence(depot, [ci, cj], current_time)
                # penalty = k * (lateness_merged - lateness_sep)
                penalty = self.time_penalty_k * max(0.0, (lateness_merged - lateness_sep))
                adjusted = base_sij - penalty
                savings.append((adjusted, base_sij, ci, cj, penalty))
        # sort by descending adjusted saving
        savings.sort(key=lambda x: x[0], reverse=True)

        # attempt merges while number of routes > num_agents
        for s_val, base_s, ci, cj, penalty in savings:
            if len(routes) <= num_agents:
                break
            ri = cust_to_route.get(ci)
            rj = cust_to_route.get(cj)
            if ri is None or rj is None or ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            # check if ci at end of route_i and cj at start of route_j (or vice versa)
            candidate = None
            if route_i[-1] == ci and route_j[0] == cj:
                candidate = route_i + route_j
            elif route_j[-1] == cj and route_i[0] == ci:
                candidate = route_j + route_i
            else:
                # not mergeable in Clarke-Wright simple form
                continue

            # capacity check (strict)
            if route_demand(candidate) > max_cap:
                continue

            # For soft time-window, we *do not* reject merges just because ETA > due.
            # Instead, we've already adjusted savings by expected lateness penalty,
            # so we accept merges based on adjusted savings while still respecting capacity.
            # Perform merge: replace the two routes by the merged one.
            new_routes = []
            for idx, r in enumerate(routes):
                if idx == ri or idx == rj:
                    continue
                new_routes.append(r)
            new_routes.append(candidate)
            routes = new_routes
            # rebuild cust_to_route
            cust_to_route = {}
            for idx, r in enumerate(routes):
                for c in r:
                    cust_to_route[c] = idx

        # After merging, we may still have more routes than vehicles. If so, we'll assign multiple routes per vehicle later.

        # Now assign routes to agents considering capacity and soft time-window (minimize lateness)
        agent_remaining = [a.s for a in agent_states]
        assignments: List[Optional[List[Tuple[int, int]]]] = [None] * num_agents

        # Sort routes by descending total demand (bigger routes assigned first)
        routes_sorted = sorted(routes, key=lambda r: route_demand(r), reverse=True)

        for r in routes_sorted:
            rd = route_demand(r)
            best_agent = -1
            best_metric = float("inf")
            best_lateness = float("inf")
            # find agent that minimizes lateness (soft), break ties by distance to first node
            for a_idx, ag in enumerate(agent_states):
                if agent_remaining[a_idx] < rd:
                    continue
                start_pos = (ag.x, ag.y)
                lateness = lateness_for_sequence(start_pos, r, current_time)
                # metric combines lateness (primary) and travel time to first node (secondary)
                metric = lateness * self.assign_lateness_weight + travel_time(start_pos, r[0])
                if metric < best_metric:
                    best_metric = metric
                    best_agent = a_idx
                    best_lateness = lateness
            if best_agent >= 0:
                assignments[best_agent] = r
                agent_remaining[best_agent] -= rd
            else:
                # no single agent can take the whole route (capacity limits)
                # split into per-customer assignment using best-agent-by-lateness heuristic
                for node in r:
                    ndemand = obs_map.get(node, {}).get("demand", 0)
                    best_a = -1
                    best_m = float("inf")
                    for a_idx, ag in enumerate(agent_states):
                        if agent_remaining[a_idx] < ndemand:
                            continue
                        start_pos = (ag.x, ag.y)
                        lateness = lateness_for_sequence(start_pos, [node], current_time)
                        metric = lateness * self.assign_lateness_weight + travel_time(start_pos, node)
                        if metric < best_m:
                            best_m = metric
                            best_a = a_idx
                    if best_a >= 0:
                        if assignments[best_a] is None:
                            assignments[best_a] = [node]
                        else:
                            assignments[best_a].append(node)
                        agent_remaining[best_a] -= ndemand
                    else:
                        # As a last resort, skip the node (it will remain unassigned and be handled by reactive insertion)
                        # This prevents infinite blocking when no capacity exists
                        continue

        # Convert assignments to deques for return; ensure each agent has at least depot if no assigned nodes
        plans: List[Deque[Target]] = []
        for a_idx in range(num_agents):
            assigned_route = assignments[a_idx]
            if assigned_route is None or len(assigned_route) == 0:
                plans.append(deque([depot]))
            else:
                plans.append(deque(assigned_route))

        return plans