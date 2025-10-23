from __future__ import annotations
from typing import Deque, List, Tuple
from collections import deque
from .base import BasePlanner, AgentState, Target


class NetPlanner(BasePlanner):
	"""Placeholder for a learned planner policy.

	Integrate with `models/planner_model/` later.
	"""

	def plan(
		self,
		observations: List[Tuple[int, int, int, int]],
		agent_states: List[AgentState],
		depot: Tuple[int, int],
		t: int,
		horizon: int = 1,
	) -> List[Deque[Target]]:
		# TODO: use a learned model to produce target queues
		return [deque() for _ in agent_states]
