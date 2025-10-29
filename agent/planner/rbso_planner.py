from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Set, Dict
from collections import deque
from .base import BasePlanner, AgentState, Target
from agent.controller.distance import travel_time
import random


class RepairBasedStabilityOptimizer(BasePlanner):
    """
    Repair-based Stability Optimizer (RBSO)
    节点数据结构: (x, y, t_arrival, demand, t_due)
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.destroy_ratio = params.get("destroy_ratio", 0.3)
        self.local_search_iters = params.get("local_search_iters", 10)

    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        # build obs_map for due times and demand
        obs_map: Dict[Tuple[int, int], Dict[str, int]] = {}
        for (x, y, t_arrival, demand, t_due) in observations:
            key = (x, y)
            prev = obs_map.get(key)
            if prev is None or t_due < prev["t_due"]:
                obs_map[key] = {"t_arrival": t_arrival, "demand": demand, "t_due": t_due}

        if current_plans is None or all(len(p) == 0 for p in current_plans):
            return self._initial_planning(observations, agent_states, depot, t, obs_map)

        current_plans = [deque(p) for p in current_plans]

        # detect new and cancelled demands
        current_nodes = set(obs_map.keys())
        planned_nodes = set()
        for p in current_plans:
            for n in p:
                planned_nodes.add(n)
        new_demands = current_nodes - planned_nodes
        cancelled_demands = planned_nodes - current_nodes

        affected_vehicles: Set[int] = set()
        # remove cancelled nodes from plans
        for vidx, p in enumerate(current_plans):
            newp = deque()
            for n in p:
                if n not in cancelled_demands:
                    newp.append(n)
                else:
                    affected_vehicles.add(vidx)
            current_plans[vidx] = newp

        # choose vehicles to repair if new_demands exist
        if new_demands:
            vehicle_loads = [(vidx, len(list(p))) for vidx, p in enumerate(current_plans)]
            vehicle_loads.sort(key=lambda x: x[1])
            num_to_affect = max(1, int(len(agent_states) * self.destroy_ratio))
            for vidx, _ in vehicle_loads[:num_to_affect]:
                affected_vehicles.add(vidx)

        # collect nodes to reassign
        nodes_to_reassign: List[Tuple[int, int]] = []
        for vidx in sorted(affected_vehicles):
            for n in list(current_plans[vidx]):
                if n != depot:
                    nodes_to_reassign.append(n)
            current_plans[vidx] = deque()

        for (x, y) in new_demands:
            nodes_to_reassign.append((x, y))

        # rebuild for affected vehicles
        if affected_vehicles and nodes_to_reassign:
            affected_agents = [agent_states[i] for i in sorted(affected_vehicles)]
            rebuilt = self._rebuild_routes(nodes_to_reassign, affected_agents, depot, t, obs_map)
            for i, vidx in enumerate(sorted(affected_vehicles)):
                current_plans[vidx] = rebuilt[i]

        # ensure each vehicle has at least depot target
        for vidx, p in enumerate(current_plans):
            if len(p) == 0:
                p.append(depot)

        return current_plans

    def _initial_planning(
        self,
        observations: List[Tuple[int, int, int, int, int]],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        current_time: int,
        obs_map: Dict[Tuple[int, int], Dict[str, int]],
    ) -> List[Deque[Target]]:
        plans = [deque() for _ in agent_states]
        available = [coord for coord, info in obs_map.items() if info["demand"] > 0]

        # greedy assign by nearest
        while available:
            for vidx, ag in enumerate(agent_states):
                if not available:
                    break
                if len(plans[vidx]) == 0:
                    curpos = (ag.x, ag.y)
                else:
                    curpos = plans[vidx][-1]
                # choose nearest that is feasible wrt due time
                best_i = None
                best_cost = float("inf")
                for i, node in enumerate(available):
                    dist = travel_time(curpos, node)
                    # simulate ETA if added
                    route_points = list(plans[vidx]) + [node]
                    etas = self._compute_etas((ag.x, ag.y), route_points, current_time)
                    feasible = True
                    for rp, eta in zip(route_points, etas):
                        info = obs_map.get(rp)
                        if info and eta > info["t_due"]:
                            feasible = False
                            break
                    if not feasible:
                        continue
                    if dist < best_cost:
                        best_cost = dist
                        best_i = i
                if best_i is not None:
                    plans[vidx].append(available.pop(best_i))
                else:
                    # no feasible node for this agent now
                    continue

        for p in plans:
            if len(p) == 0:
                p.append(depot)
        return plans

    def _rebuild_routes(
        self,
        nodes: List[Tuple[int, int]],
        agents: List[AgentState],
        depot: Tuple[int, int],
        current_time: int,
        obs_map: Dict[Tuple[int, int], Dict[str, int]],
    ) -> List[Deque[Target]]:
        # greedily insert nodes into affected vehicles using insertion-cost with time-window checks
        plans = [deque() for _ in agents]
        remaining = nodes.copy()

        while remaining:
            best = {"cost": float("inf"), "node": None, "veh": -1, "pos": -1}
            for node in remaining:
                for vidx, ag in enumerate(agents):
                    # simple capacity check (assume each node demand is obs_map[node]['demand'])
                    node_demand = obs_map.get(node, {}).get("demand", 1)
                    if ag.s < node_demand:
                        continue
                    route = [(ag.x, ag.y)] + list(plans[vidx])
                    # try insert positions
                    for pos in range(1, len(route) + 1):
                        prev = route[pos - 1]
                        if pos < len(route):
                            nxt = route[pos]
                            orig = travel_time(prev, nxt)
                            newd = travel_time(prev, node) + travel_time(node, nxt)
                            delta = newd - orig
                        else:
                            last = route[-1]
                            delta = travel_time(last, node)
                        # simulate plan after insertion
                        new_plan_points = list(plans[vidx])
                        insert_idx = pos - 1
                        new_plan_points.insert(insert_idx, node)
                        etas = self._compute_etas((ag.x, ag.y), new_plan_points, current_time)
                        feasible = True
                        for rp, eta in zip(new_plan_points, etas):
                            info = obs_map.get(rp)
                            if info and eta > info["t_due"]:
                                feasible = False
                                break
                        if not feasible:
                            continue
                        if delta < best["cost"]:
                            best = {"cost": delta, "node": node, "veh": vidx, "pos": insert_idx}
            if best["node"] is None:
                break
            # insert
            vplan = list(plans[best["veh"]])
            if best["pos"] >= len(vplan):
                vplan.append(best["node"])
            else:
                vplan.insert(best["pos"], best["node"])
            plans[best["veh"]] = deque(vplan)
            remaining.remove(best["node"])

        # ensure each plan non-empty
        for i in range(len(plans)):
            if len(plans[i]) == 0:
                plans[i].append(depot)
        return plans

    def _compute_etas(self, start_pos: Tuple[int, int], route_points: List[Tuple[int, int]], start_time: int) -> List[int]:
        etas = []
        cur = start_pos
        cur_time = start_time
        for p in route_points:
            cur_time += travel_time(cur, p)
            etas.append(cur_time)
            cur = p
        return etas