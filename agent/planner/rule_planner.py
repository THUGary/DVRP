from __future__ import annotations
from typing import Deque, List, Tuple
from collections import deque
from .base import BasePlanner, AgentState, Target


class RuleBasedPlanner(BasePlanner):
	"""Greedy planner with unique assignment per step.

	Assign agents one by one. For each agent, pick the nearest available demand
	(not yet assigned in this planning call). If no demand remains, send to depot.
	Returns a queue (length 1) per agent.
	"""

	def plan(
		self,
		observations: List[Tuple[int, int, int, int]],
		agent_states: List[AgentState],
		depot: Tuple[int, int],
		t: int,
		horizon: int = 1,
	) -> List[Deque[Target]]:
		# Build unique set of demand coordinates only (ignore duplicates on same cell)
		# Compatible with obs tuple length 4 or 5: (x,y,t,c) or (x,y,t,c,end_t)
		available: List[Target] = []
		seen = set()
		for d in observations:
			# unpack first 4 fields
			x, y, _t, c = d[:4]
			if c <= 0:
				continue
			key = (x, y)
			if key not in seen:
				seen.add(key)
				available.append(key)

		out: List[Deque[Target]] = []
		for a in agent_states:
			if available:
				idx, (tx, ty) = min(
					enumerate(available), key=lambda it: abs(it[1][0] - a.x) + abs(it[1][1] - a.y)
				)
				out.append(deque([(tx, ty)]))
				# Remove the selected target so it won't be selected again
				available.pop(idx)
			else:
				out.append(deque([depot]))
		return out
