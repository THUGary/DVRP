from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Dict
from collections import deque
from .base import BasePlanner, AgentState, Target
from agent.controller.distance import travel_time


class DistributedCooperativePlanner(BasePlanner):
    """
    Distributed Cooperative Planner (DCP) - 修改版

    变更要求实现：
    1) 将时间窗的硬约束改为软约束：在计算出价（bid）时加入线性惩罚项（lateness 增量 * time_penalty_k），
       不再因为时间窗不满足而直接不出价；
    2) 每次针对当前的未服务节点重新进行拍卖规划（从头规划），**忽略** current_plans 参数（即不沿用之前的路径）。
       也就是说每次计划是基于当前未服务节点和车辆当前位置的全新拍卖分配。

    节点数据结构: (x, y, t_arrival, demand, t_due)
    Planner 参数（通过 constructor params 传入）:
      - auction_rounds: int         (默认 3)
      - time_penalty_k: float       (线性时间惩罚系数，默认 1.0)
      - bid_strategy: str           ('distance' | 'time_urgency')  (用于计算基础出价)
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.auction_rounds: int = int(params.get("auction_rounds", 3))
        self.time_penalty_k: float = float(params.get("time_penalty_k", 1.0))
        self.bid_strategy: str = params.get("bid_strategy", "time_urgency")

    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,  # IGNORED by this planner (fresh auction each call)
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        # Build obs_map keyed by coordinate, keep most urgent (earliest due) if duplicates
        obs_map: Dict[Tuple[int, int], Dict[str, int]] = {}
        for (x, y, t_arrival, demand, t_due) in observations:
            key = (x, y)
            prev = obs_map.get(key)
            if prev is None or t_due < prev["t_due"]:
                obs_map[key] = {"t_arrival": t_arrival, "demand": demand, "t_due": t_due}

        # Unassigned demands: all currently observed demands (we ignore current_plans)
        unassigned: List[Tuple[int, int, int, int, int]] = []
        for (coord, info) in obs_map.items():
            if info["demand"] > 0:
                unassigned.append((coord[0], coord[1], info["demand"], info["t_arrival"], info["t_due"]))

        num_agents = len(agent_states)
        # tentative plans built fresh by auction; start empty for each agent
        tentative_plans: List[Deque[Target]] = [deque() for _ in agent_states]
        # available capacities (remaining) per agent (we deduct demand when assigning)
        agent_remaining = [ag.s for ag in agent_states]

        # helper: compute ETAs for a route_points list starting from start_pos at start_time
        def compute_etas(start_pos: Tuple[int, int], route_points: List[Tuple[int, int]], start_time: int) -> List[int]:
            etas: List[int] = []
            cur = start_pos
            cur_time = start_time
            for p in route_points:
                cur_time += travel_time(cur, p)
                etas.append(cur_time)
                cur = p
            return etas

        # helper: compute lateness sum for route (sum positive (ETA - t_due))
        def lateness_sum_for_route(start_pos: Tuple[int, int], route_points: List[Tuple[int, int]], start_time: int) -> float:
            etas = compute_etas(start_pos, route_points, start_time)
            s = 0.0
            for rp, eta in zip(route_points, etas):
                info = obs_map.get(rp)
                if info:
                    s += max(0.0, eta - info["t_due"])
            return s

        # Auction rounds: repeatedly collect bids and allocate
        for _round in range(self.auction_rounds):
            if not unassigned:
                break

            # collect bids: dict node_coord -> list of (vehicle_idx, bid_value, best_insert_pos, delta_travel, delta_lateness)
            bids: Dict[Tuple[int, int], List[Tuple[int, float, int, float, float]]] = {}

            for (x, y, demand, t_arrival, t_due) in list(unassigned):
                node = (x, y)
                bids[node] = []

                for vidx, ag in enumerate(agent_states):
                    # capacity check: strict (cannot assign if remaining capacity insufficient)
                    if agent_remaining[vidx] < demand:
                        continue

                    # build route for this agent as current tentative plan
                    route_points = list(tentative_plans[vidx])
                    route = [(ag.x, ag.y)] + route_points  # route with vehicle pos at index 0

                    best_cost = float("inf")
                    best_pos = 0
                    best_delta_travel = 0.0
                    best_delta_lateness = 0.0

                    # compute current lateness for agent's tentative route
                    current_lateness = lateness_sum_for_route((ag.x, ag.y), route_points, t)

                    # Try inserting node at each position between route elements
                    for pos in range(1, len(route) + 1):
                        prev = route[pos - 1]
                        if pos < len(route):
                            nxt = route[pos]
                            original = travel_time(prev, nxt)
                            newd = travel_time(prev, node) + travel_time(node, nxt)
                            delta_travel = newd - original
                        else:
                            # insert at end: cost = distance from last to node
                            last = route[-1]
                            delta_travel = travel_time(last, node)

                        # simulate new plan points and compute lateness after insertion
                        insert_idx = pos - 1  # index in tentative plan list
                        new_plan_points = route_points.copy()
                        new_plan_points.insert(insert_idx, node)
                        new_lateness = lateness_sum_for_route((ag.x, ag.y), new_plan_points, t)
                        delta_lateness = new_lateness - current_lateness

                        # base bid value depends on strategy (here base is delta_travel)
                        if self.bid_strategy == "distance":
                            base_bid = float(delta_travel)
                        elif self.bid_strategy == "time_urgency":
                            # less cost if node is more urgent (small slack)
                            slack = max(1.0, t_due - t)
                            base_bid = float(delta_travel) / slack
                        else:
                            base_bid = float(delta_travel)

                        # final bid: base + time_penalty_k * delta_lateness (linear penalty, soft time windows)
                        bid_value = base_bid + (self.time_penalty_k * delta_lateness)

                        # store best pos for this agent
                        if bid_value < best_cost:
                            best_cost = bid_value
                            best_pos = insert_idx
                            best_delta_travel = delta_travel
                            best_delta_lateness = delta_lateness

                    # After trying all insert positions, if we found a finite best_cost, append bid
                    if best_cost < float("inf"):
                        bids[node].append((vidx, best_cost, best_pos, best_delta_travel, best_delta_lateness))

            # Allocation: allocate each node to the vehicle with lowest bid (simple greedy auction)
            # We process nodes sorted by their minimum bid (most contested / cheapest first)
            node_items = []
            for node, blist in bids.items():
                if blist:
                    min_bid = min(b[1] for b in blist)
                else:
                    min_bid = float("inf")
                node_items.append((node, min_bid, blist))
            node_items.sort(key=lambda it: it[1])  # ascending by min bid

            newly_assigned = set()
            for node, _min_bid, blist in node_items:
                if not blist:
                    continue
                # choose the vehicle with minimum bid for this node
                blist.sort(key=lambda x: x[1])  # sort by bid_value
                for (vidx, bid_value, insert_pos, delta_travel, delta_lateness) in blist:
                    # ensure capacity still holds (may have changed during this round due to other allocations)
                    # demand value:
                    demand = obs_map.get(node, {}).get("demand", 0)
                    if agent_remaining[vidx] < demand:
                        continue
                    # assign node to this vehicle
                    plan_list = list(tentative_plans[vidx])
                    if insert_pos >= len(plan_list):
                        plan_list.append(node)
                    else:
                        plan_list.insert(insert_pos, node)
                    tentative_plans[vidx] = deque(plan_list)
                    agent_remaining[vidx] -= demand
                    newly_assigned.add(node)
                    break  # move to next node

            # remove newly assigned nodes from unassigned
            if newly_assigned:
                unassigned = [d for d in unassigned if (d[0], d[1]) not in newly_assigned]

        # End auction rounds. Ensure each vehicle has at least depot target
        final_plans: List[Deque[Target]] = []
        for idx, p in enumerate(tentative_plans):
            if len(p) == 0:
                final_plans.append(deque([depot]))
            else:
                final_plans.append(deque(p))

        return final_plans