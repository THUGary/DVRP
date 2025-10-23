from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from agent.planner.base import AgentState
from agent.generator.base import BaseDemandGenerator, Demand


Action = Tuple[int, int]  # (dx, dy) per agent


@dataclass
class EnvState:
	time: int
	agent_states: List[AgentState]
	depot: Tuple[int, int]
	demands: List[Demand] = field(default_factory=list)


class GridEnvironment:
	"""Grid world DVRP environment skeleton.

	State = {agents + depot + current demands}. Action is per-agent (dx, dy) with unit moves.
	"""

	def __init__(
		self,
		width: int,
		height: int,
		num_agents: int,
		capacity: int,
		depot: Tuple[int, int] = (0, 0),
		generator: Optional[BaseDemandGenerator] = None,
		max_time: int = 100,
	) -> None:
		self.width = width
		self.height = height
		self.max_time = max_time
		self.depot = depot
		self.capacity = capacity
		self._generator = generator
		self._state: Optional[EnvState] = None

	# --- Core API ---
	def reset(self, seed: Optional[int] = None) -> Dict:
		if self._generator:
			self._generator.reset(seed)
		agent_states = [AgentState(x=self.depot[0], y=self.depot[1], s=self.capacity) for _ in range(self._num_agents)] if hasattr(self, "_num_agents") else []
		# If num_agents was not set via property yet, default to 1
		if not agent_states:
			self._num_agents = 1
			agent_states = [AgentState(x=self.depot[0], y=self.depot[1], s=self.capacity)]
		self._state = EnvState(time=0, agent_states=agent_states, depot=self.depot, demands=[])
		return self._obs()

	def step(self, actions: List[Action]) -> Tuple[Dict, float, bool, Dict]:

		assert self._state is not None, "Call reset() first"
		t = self._state.time
		# 1) new demands appear
		if self._generator:
			new_demands = self._generator.sample(t)
			self._state.demands.extend(new_demands)
		# 1.5) cancel expired demands (strictly after end_t)
		self._state.demands = [d for d in self._state.demands if t <= d.end_t]
		# 2) apply actions to agents
		reward = 0.0
		for i, (dx, dy) in enumerate(actions):
			a = self._state.agent_states[i]
			nx = max(0, min(self.width - 1, a.x + max(-1, min(1, dx))))
			ny = max(0, min(self.height - 1, a.y + max(-1, min(1, dy))))
			self._state.agent_states[i] = AgentState(x=nx, y=ny, s=a.s)
		# 3) serve demands when an agent arrives on a demand cell (simplified: remove demand entirely)
		remaining: List[Demand] = []
		pos_to_agent_idx = {(a.x, a.y): idx for idx, a in enumerate(self._state.agent_states)}
		for d in self._state.demands:
			agent_idx = pos_to_agent_idx.get((d.x, d.y))
			if agent_idx is not None:
				# agent on the demand cell
				a = self._state.agent_states[agent_idx]
				# Assumption: must have enough capacity to serve; otherwise skip
				if a.s >= d.c:
					# serve
					new_s = a.s - d.c
					self._state.agent_states[agent_idx] = AgentState(x=a.x, y=a.y, s=new_s)
					reward += 1.0
					# drop demand (served)
					continue
			# keep demand (not served)
			remaining.append(d)
		#print agents' capacity
		for i, a in enumerate(self._state.agent_states):
			print(f"Agent {i} capacity: {a.s}")
		self._state.demands = remaining
		# 4) time update
		self._state.time += 1
		done = self._state.time >= self.max_time
		return self._obs(), reward, done, {}

	# --- Helpers ---
	def _obs(self) -> Dict:
		assert self._state is not None
		return {
			"time": self._state.time,
			"depot": self._state.depot,
			"agent_states": [(a.x, a.y, a.s) for a in self._state.agent_states],
			"demands": [(d.x, d.y, d.t, d.c, d.end_t) for d in self._state.demands if d.t <= self._state.time],
			"width": self.width,
			"height": self.height,
		}

	# Property to set/get number of agents post-init
	@property
	def num_agents(self) -> int:
		return getattr(self, "_num_agents", len(self._state.agent_states) if self._state else 0)

	@num_agents.setter
	def num_agents(self, n: int) -> None:
		self._num_agents = n
