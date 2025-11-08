from __future__ import annotations
from typing import Deque, List, Tuple, Optional
from collections import deque
from .base import BasePlanner, AgentState, Target


class NetPlanner(BasePlanner):
	"""Placeholder for a learned planner policy.

	Integrate with `models/planner_model/` later.
	
	节点数据结构: (x, y, t_arrival, t_due, demand)
	"""

	def plan(
		self,
		observations: List[Tuple[int, int, int, int, int]],  # [(x, y, t_arrival, c, t_due), ...]
		agent_states: List[AgentState],
		depot: Tuple[int, int],
		t: int,
		horizon: int = 1,
		current_plans: Optional[List[Deque[Target]]] = None,
		global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,  # [(x, y, t_arrival, t_due, demand), ...]
		serve_mark: Optional[List[int]] = None,
		unserved_count: Optional[int] = None,
	) -> List[Deque[Target]]:
		# TODO: use a learned model to produce target queues
		# 可以使用 current_plans, global_nodes, serve_mark, unserved_count 等信息
		# global_nodes 中每个节点包含: (x, y, t_arrival, t_due, demand)
		return [deque() for _ in agent_states]