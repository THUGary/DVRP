from __future__ import annotations
from typing import Deque, List, Tuple, Optional
from collections import deque
from .base import BasePlanner, AgentState, Target


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
	"""

	def __init__(self, full_capacity: int | None = None) -> None:
		if full_capacity is None:
			raise RuntimeError("RuleBasedPlanner requires full_capacity (Config.capacity); none provided.")
		self.full_capacity = int(full_capacity)

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
				# no more demands; send all agents to depot for remaining steps
				for i in range(A):
					out[i].append(depot)
					cur_pos[i] = depot
					# restore capacity when at depot
					cur_cap[i] = full_cap[i]
				continue

			# Assign greedily this step, ensuring uniqueness
			used_indices: set[int] = set()
			for i in range(A):
				if not available_xy:
					out[i].append(depot)
					cur_pos[i] = depot
					cur_cap[i] = full_cap[i]
					continue
				# choose nearest available target to current position
				cx, cy = cur_pos[i]
				best_j = None
				best_d = None
				for j, (tx, ty) in enumerate(available_xy):
					if j in used_indices:
						continue
					# capacity feasibility: require capacity >= demand
					req = available_dem[j]
					if not (cur_cap[i] >= req):
						continue
					d = abs(tx - cx) + abs(ty - cy)
					if best_d is None or d < best_d:
						best_d = d
						best_j = j
				if best_j is None:
					# no feasible targets left this step
					out[i].append(depot)
					cur_pos[i] = depot
					cur_cap[i] = full_cap[i]
					continue
				# assign and mark as used; remove from pool permanently after this step
				tx, ty = available_xy[best_j]
				out[i].append((tx, ty))
				cur_pos[i] = (tx, ty)
				# update capacity strictly reducing by demand
				cur_cap[i] = max(0, cur_cap[i] - available_dem[best_j])
				used_indices.add(best_j)

			# remove used targets from available (in descending index order)
			if used_indices:
				for j in sorted(used_indices, reverse=True):
					available_xy.pop(j)
					available_dem.pop(j)

		return out