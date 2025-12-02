from __future__ import annotations
from typing import Deque, List, Tuple, Optional
from collections import deque
from .base import BasePlanner, AgentState, Target
from .exact_solver import ExactVRPSolver


class RuleBasedPlanner(BasePlanner):
	"""Greedy planner with unique assignment per step and capacity feasibility.

	Assign agents one by one. At each planning step, for each agent, pick the nearest
	available demand whose demand is feasible under remaining capacity (capacity >= demand).
	Ensure uniqueness: a node is assigned to at most one agent in the same step, and once
	assigned it is removed from the pool for subsequent steps. If no feasible demand remains
	for an agent in a step, send this agent to depot for that step (restoring its capacity
	to full). If globally no demands remain, send all agents to depot for the remaining steps.
	Returns a queue per agent with up to `horizon` targets.
	
	节点数据结构: (x, y, t_arrival, c, t_due) 其中 c 表示该节点需求量

	索引约定（用于与模型/数据的标签对齐）:
	- 模型与数据标签中，depot 的类别索引固定为 0
	- nodes 的类别索引为 1..N，并与 nodes 列表中的下标 0..N-1 一一对应（i -> i+1）
	- 本规划器仅返回目标坐标，不产生索引；索引映射由上层数据拼装（data_gen）来完成
	
	Modes:
	- greedy: Nearest-neighbor heuristic (fast, approximate)
	- optimize: Hungarian assignment (better, O(n³))
	- exact: Optimal DP-based solution (best, exponential - warns if >12 nodes but still uses DP)
	- heuristic: High-quality heuristic (Clarke-Wright + local search, good for larger instances)
	"""
	
	# Warning threshold for exact mode
	EXACT_WARN_NODES = 12  # Warn if node count exceeds this

	def __init__(self, full_capacity: int | None = None, *, mode: str = "greedy") -> None:
		if full_capacity is None:
			raise RuntimeError("RuleBasedPlanner requires full_capacity (Config.capacity); none provided.")
		mode_normalized = (mode or "greedy").lower()
		if mode_normalized not in {"greedy", "optimize", "exact", "heuristic"}:
			raise ValueError(f"Unsupported rule-based planner mode '{mode}'. Use 'greedy', 'optimize', 'exact', or 'heuristic'.")
		self.full_capacity = int(full_capacity)
		self.mode = mode_normalized
		self._large_cost = 1e6
		self._exact_solver: Optional[ExactVRPSolver] = None
		self._cached_exact_routes: Optional[List[deque]] = None
		self._cached_exact_demands_hash: Optional[int] = None

	def plan(
		self,
		observations: List[Tuple[int, int, int, int, int]],  # [(x, y, t_arrival, c, t_due), ...]
		agent_states: List[AgentState],
		depot: Tuple[int, int],
		t: int,
		horizon: int = 1,
		current_plans: Optional[List[Deque[Target]]] = None,
		global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
		serve_mark: Optional[List[int]] = None,
		unserved_count: Optional[int] = None,
	) -> List[Deque[Target]]:
		
		# Build unique set of demand coordinates with demand (ignore duplicates on same cell)
		# Keep the first occurrence's demand value
		available_xy: List[Target] = []
		available_dem: List[int] = []
		seen = set()
		for (x, y, t_arrival, c, t_due) in observations:
			if c <= 0:
				continue
			key = (x, y)
			if key not in seen:
				seen.add(key)
				available_xy.append(key)
				available_dem.append(int(c))

		# Initialize per-agent output queues and current positions
		A = len(agent_states)
		
		# EXACT mode: compute optimal solution once and cache it
		if self.mode == "exact":
			return self._plan_exact(available_xy, available_dem, agent_states, depot, current_plans, force_dp=True)
		
		# HEURISTIC mode: use high-quality heuristic (Clarke-Wright + local search)
		if self.mode == "heuristic":
			return self._plan_exact(available_xy, available_dem, agent_states, depot, current_plans, force_dp=False)
		
		out: List[Deque[Target]] = [deque() for _ in range(A)]
		# snapshot agent states (copy values instead of referencing AgentState objects)
		snapshot: List[Tuple[int, int, int]] = [(int(a.x), int(a.y), int(a.s)) for a in agent_states]
		cur_pos: List[Tuple[int, int]] = [(x, y) for (x, y, _s) in snapshot]
		cur_cap: List[int] = [s for (_x, _y, s) in snapshot]
		# 满容量必须由构造时提供的 full_capacity 指定
		full_cap: List[int] = [int(self.full_capacity) for _ in agent_states]

		steps = max(1, int(horizon))
		for _step in range(steps):
			if not available_xy:
				self._send_all_to_depot(out, cur_pos, cur_cap, full_cap, depot)
				continue

			if self.mode == "optimize":
				assignments = self._optimize_assignments(cur_pos, cur_cap, available_xy, available_dem, depot)
				used_indices: set[int] = set()
				for agent_idx, demand_idx in enumerate(assignments):
					if demand_idx is None:
						self._send_agent_to_depot(agent_idx, out, cur_pos, cur_cap, full_cap, depot)
						continue
					req = available_dem[demand_idx]
					if cur_cap[agent_idx] < req:
						self._send_agent_to_depot(agent_idx, out, cur_pos, cur_cap, full_cap, depot)
						continue
					tx, ty = available_xy[demand_idx]
					out[agent_idx].append((tx, ty))
					cur_pos[agent_idx] = (tx, ty)
					cur_cap[agent_idx] = max(0, cur_cap[agent_idx] - req)
					used_indices.add(demand_idx)
			else:
				used_indices = set()
				for i in range(A):
					if not available_xy:
						self._send_agent_to_depot(i, out, cur_pos, cur_cap, full_cap, depot)
						continue
					best_j = self._nearest_feasible_target(cur_pos[i], cur_cap[i], available_xy, available_dem, used_indices)
					if best_j is None:
						self._send_agent_to_depot(i, out, cur_pos, cur_cap, full_cap, depot)
						continue
					tx, ty = available_xy[best_j]
					out[i].append((tx, ty))
					cur_pos[i] = (tx, ty)
					cur_cap[i] = max(0, cur_cap[i] - available_dem[best_j])
					used_indices.add(best_j)

			if used_indices:
				for j in sorted(used_indices, reverse=True):
					available_xy.pop(j)
					available_dem.pop(j)

		return out

	def _send_agent_to_depot(
		self,
		agent_idx: int,
		out: List[Deque[Target]],
		cur_pos: List[Tuple[int, int]],
		cur_cap: List[int],
		full_cap: List[int],
		depot: Tuple[int, int],
	) -> None:
		out[agent_idx].append(depot)
		cur_pos[agent_idx] = depot
		cur_cap[agent_idx] = full_cap[agent_idx]

	def _send_all_to_depot(
		self,
		out: List[Deque[Target]],
		cur_pos: List[Tuple[int, int]],
		cur_cap: List[int],
		full_cap: List[int],
		depot: Tuple[int, int],
	) -> None:
		for i in range(len(out)):
			self._send_agent_to_depot(i, out, cur_pos, cur_cap, full_cap, depot)

	def _nearest_feasible_target(
		self,
		agent_pos: Tuple[int, int],
		agent_cap: int,
		available_xy: List[Target],
		available_dem: List[int],
		used_indices: set[int],
	) -> Optional[int]:
		cx, cy = agent_pos
		best_j = None
		best_d = None
		for j, (tx, ty) in enumerate(available_xy):
			if j in used_indices:
				continue
			req = available_dem[j]
			if agent_cap < req:
				continue
			d = abs(tx - cx) + abs(ty - cy)
			if best_d is None or d < best_d:
				best_d = d
				best_j = j
		return best_j

	def _optimize_assignments(
		self,
		cur_pos: List[Tuple[int, int]],
		cur_cap: List[int],
		available_xy: List[Target],
		available_dem: List[int],
		depot: Tuple[int, int],
	) -> List[Optional[int]]:
		A = len(cur_pos)
		D = len(available_xy)
		if A == 0:
			return []
		if D == 0:
			return [None for _ in range(A)]
		size = max(A, D)
		cost_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
		for i in range(size):
			for j in range(size):
				if i < A and j < D:
					req = available_dem[j]
					if cur_cap[i] >= req:
						cx, cy = cur_pos[i]
						tx, ty = available_xy[j]
						cost_matrix[i][j] = abs(tx - cx) + abs(ty - cy)
					else:
						cost_matrix[i][j] = self._large_cost
				elif i < A and j >= D:
					cx, cy = cur_pos[i]
					cost_matrix[i][j] = abs(depot[0] - cx) + abs(depot[1] - cy)
				else:
					cost_matrix[i][j] = 0.0
		assignment_cols = self._hungarian(cost_matrix)
		result: List[Optional[int]] = [None for _ in range(A)]
		for i in range(min(A, len(assignment_cols))):
			j = assignment_cols[i]
			if j is None or j >= D:
				result[i] = None
				continue
			if cost_matrix[i][j] >= self._large_cost / 2:
				result[i] = None
				continue
			result[i] = j
		return result

	def _hungarian(self, cost: List[List[float]]) -> List[Optional[int]]:
		n = len(cost)
		if n == 0:
			return []
		u = [0.0] * (n + 1)
		v = [0.0] * (n + 1)
		p = [0] * (n + 1)
		way = [0] * (n + 1)
		for i in range(1, n + 1):
			p[0] = i
			minv = [float("inf")] * (n + 1)
			used = [False] * (n + 1)
			j0 = 0
			while True:
				used[j0] = True
				i0 = p[j0]
				delta = float("inf")
				j1 = 0
				for j in range(1, n + 1):
					if used[j]:
						continue
					cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
					if cur < minv[j]:
						minv[j] = cur
						way[j] = j0
					if minv[j] < delta:
						delta = minv[j]
						j1 = j
				for j in range(0, n + 1):
					if used[j]:
						u[p[j]] += delta
						v[j] -= delta
					else:
						minv[j] -= delta
				j0 = j1
				if p[j0] == 0:
					break
			while True:
				j1 = way[j0]
				p[j0] = p[j1]
				j0 = j1
				if j0 == 0:
					break
		assignment = [None for _ in range(n)]
		for j in range(1, n + 1):
			if p[j] != 0:
				assignment[p[j] - 1] = j - 1
		return assignment

	def _plan_exact(
		self,
		available_xy: List[Target],
		available_dem: List[int],
		agent_states: List[AgentState],
		depot: Tuple[int, int],
		current_plans: Optional[List[Deque[Target]]],
		force_dp: bool = True,
	) -> List[Deque[Target]]:
		"""
		Plan using exact DP solver or heuristic for VRP solution.
		
		Args:
			force_dp: If True (exact mode), always use DP even for large instances (with warning).
			          If False (heuristic mode), always use heuristic.
		
		For static demands, this computes the routes once and caches them.
		Subsequent calls return the cached routes (with served nodes already removed by controller).
		"""
		A = len(agent_states)
		
		# If no demands, return depot for all agents
		if not available_xy:
			return [deque([depot]) for _ in range(A)]
		
		# If we have already computed solution, return current plans
		# (controller consumes targets via popleft, so current_plans tracks progress)
		if self._cached_exact_routes is not None and current_plans is not None:
			# Return current plans - they are updated by the controller as nodes are visited
			return list(current_plans)
		
		n = len(available_xy)
		
		# Warn for large instances in exact mode
		if force_dp and n > self.EXACT_WARN_NODES:
			import sys
			print(f"[WARNING] Exact mode with {n} nodes (>{self.EXACT_WARN_NODES}). "
				  f"This may take a long time. Consider using 'heuristic' mode.", file=sys.stderr)
		
		# Compute new solution (only on first call)
		if self._exact_solver is None:
			self._exact_solver = ExactVRPSolver(
				capacity=self.full_capacity,
				num_vehicles=A,
				use_euclidean=False,  # Use Manhattan distance (agents move in 4 directions only)
			)
		
		# Solve VRP - force_dp determines exact DP vs heuristic
		total_dist, routes = self._exact_solver.solve_with_mode(
			depot=depot,
			nodes=list(available_xy),
			demands=list(available_dem),
			time_limit=60.0 if force_dp else 10.0,
			force_dp=force_dp,
		)
		
		# Convert routes to target queues, inserting depot returns when capacity is exhausted
		result: List[Deque[Target]] = []
		for v in range(A):
			targets: Deque[Target] = deque()
			if v < len(routes) and routes[v]:
				current_load = 0
				for node_idx in routes[v]:
					if node_idx < len(available_xy):
						node_demand = available_dem[node_idx]
						# Check if we need to return to depot for capacity
						if current_load + node_demand > self.full_capacity:
							# Return to depot first
							targets.append(depot)
							current_load = 0
						targets.append(available_xy[node_idx])
						current_load += node_demand
				# Add depot at the end
				targets.append(depot)
			else:
				# Idle vehicle stays at depot
				targets.append(depot)
			result.append(targets)
		
		# Cache the solution
		self._cached_exact_routes = result
		self._cached_exact_demands_hash = hash((tuple(available_xy), tuple(available_dem), depot))
		
		return result