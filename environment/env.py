from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from agent.planner.base import AgentState
from agent.generator.base import BaseDemandGenerator, Demand
import math


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
		expiry_penalty_scale: float = 1.0,
	) -> None:
		self.width = width
		self.height = height
		self.max_time = max_time
		self.depot = depot
		self.capacity = capacity
		self._generator = generator
		self.expiry_penalty_scale = expiry_penalty_scale
		self._state: Optional[EnvState] = None

		# cache for resolved full capacity to avoid repeated imports
		self._resolved_full_capacity: Optional[int] = None

	def _full_capacity(self) -> int:
		"""Return the vehicle full capacity.

		Priority:
		1) self.capacity if available
		2) configs.capacity (from project-level configs.py) as a fallback
		"""
		# Fast path: explicitly provided on init
		if getattr(self, "capacity", None) is not None:
			return int(self.capacity)
		# Cached fallback if already resolved once
		if self._resolved_full_capacity is not None:
			return int(self._resolved_full_capacity)
		# Try to import from configs.py
		try:
			import configs  # type: ignore
			value = int(getattr(configs, "capacity"))
			self._resolved_full_capacity = value
			return value
		except Exception:
			# Last-resort default to 0 to avoid crashes; callers should handle if needed
			self._resolved_full_capacity = 0
			return 0

	# --- Core API ---
	def reset(self, seed: Optional[int] = None) -> Dict:
		if self._generator:
			self._generator.reset(seed)
		agent_states = [AgentState(x=self.depot[0], y=self.depot[1], s=self._full_capacity()) for _ in range(self._num_agents)] if hasattr(self, "_num_agents") else []
		# If num_agents was not set via property yet, default to 1
		if not agent_states:
			self._num_agents = 1
			agent_states = [AgentState(x=self.depot[0], y=self.depot[1], s=self._full_capacity())]
		self._state = EnvState(time=0, agent_states=agent_states, depot=self.depot, demands=[])
		# initialize episode-level statistics
		self._episode_stats = {
			"seen_demands_ids": set(),
			"demand_count": 0,
			"demand_capacity": 0.0,
			"served_count": 0,
			"served_capacity": 0.0,
			"served_details": [],
			"agent_total_distances": [0.0 for _ in agent_states],
			"total_distance": 0.0,
			"episode_reward": 0.0,
		}
		return self._obs()

	def step(self, actions: List[Action]) -> Tuple[Dict, float, bool, Dict]:

		assert self._state is not None, "Call reset() first"
		t = self._state.time
		# 1) new demands appear
		if self._generator:
			new_demands = self._generator.sample(t)
			self._state.demands.extend(new_demands)
		else:
			new_demands = []
		# 1.5) detect and remove expired demands (strictly after end_t)
		# expired demands are those with end_t < t (they should have been handled before this step)
		expired = [d for d in self._state.demands if d.end_t < t]
		expired_count = len(expired)
		expired_capacity = sum(float(d.c) for d in expired)
		# remove expired demands
		self._state.demands = [d for d in self._state.demands if t <= d.end_t]

		# apply expiry penalty (negative), scaled by expiry_penalty_scale
		expiry_penalty = - float(self.expiry_penalty_scale) * expired_capacity if expired_count > 0 else 0.0

		# record any newly observed demands (covers both generated and externally appended demands)
		for d in self._state.demands:
			if id(d) not in self._episode_stats["seen_demands_ids"]:
				self._episode_stats["seen_demands_ids"].add(id(d))
				self._episode_stats["demand_count"] += 1
				self._episode_stats["demand_capacity"] += float(d.c)
		# 2) apply actions to agents
		# record previous positions to compute route distance
		prev_positions = [(a.x, a.y) for a in self._state.agent_states]
		capacity_reward = 0.0
		for i, (dx, dy) in enumerate(actions):
			a = self._state.agent_states[i]
			nx = max(0, min(self.width - 1, a.x + max(-1, min(1, dx))))
			ny = max(0, min(self.height - 1, a.y + max(-1, min(1, dy))))
			# if agent arrives at depot, refill to full capacity
			new_s = a.s
			if (nx, ny) == self.depot:
				new_s = self._full_capacity()
			self._state.agent_states[i] = AgentState(x=nx, y=ny, s=new_s)
		# 3) serve demands when an agent arrives on a demand cell (simplified: remove demand entirely)
		remaining: List[Demand] = []
		pos_to_agent_idx = {(a.x, a.y): idx for idx, a in enumerate(self._state.agent_states)}

		# stats before serving
		total_demands_before = len(self._state.demands)
		total_capacity_before = sum(float(d.c) for d in self._state.demands)

		served_count = 0
		served_capacity = 0.0
		served_details: List[Tuple[int, int, float]] = []
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
					# accumulate capacity served this step
					capacity_reward += float(d.c)
					# record served stats
					served_count += 1
					served_capacity += float(d.c)
					served_details.append((d.x, d.y, float(d.c)))
					# drop demand (served)
					continue
			# keep demand (not served)
			remaining.append(d)

		self._state.demands = remaining
		# compute route distance (Euclidean) this step
		movement_distance = 0.0
		agent_distances: List[float] = []
		for idx, a in enumerate(self._state.agent_states):
			px, py = prev_positions[idx]
			d = math.hypot(a.x - px, a.y - py)
			agent_distances.append(d)
			movement_distance += d

		# net reward = total capacity served this step - distance traveled this step
		# include expiry penalty (negative) for demands that timed out at the start of this step
		reward = capacity_reward - float(movement_distance) + expiry_penalty

		# update episode-level stats
		self._episode_stats["served_count"] += served_count
		self._episode_stats["served_capacity"] += served_capacity
		self._episode_stats["served_details"].extend(served_details)
		# update expired stats in episode-level tracking
		if expired_count > 0:
			self._episode_stats.setdefault('expired_count', 0)
			self._episode_stats.setdefault('expired_capacity', 0.0)
			self._episode_stats['expired_count'] += expired_count
			self._episode_stats['expired_capacity'] += expired_capacity
			self._episode_stats.setdefault('expired_penalty', 0.0)
			self._episode_stats['expired_penalty'] += -expiry_penalty  # store positive value for capacity, penalty stored as positive magnitude
		for idx, d in enumerate(agent_distances):
			# ensure agent_total_distances is large enough (in case num_agents was set after reset)
			if idx >= len(self._episode_stats["agent_total_distances"]):
				self._episode_stats["agent_total_distances"].extend([0] * (idx + 1 - len(self._episode_stats["agent_total_distances"])))
			self._episode_stats["agent_total_distances"][idx] += d
		self._episode_stats["total_distance"] += movement_distance
		self._episode_stats["episode_reward"] += reward

		# 4) time update
		self._state.time += 1
		done = self._state.time >= self.max_time
		# on episode end, print aggregated statistics and return them in info
		info: Dict = {}
		if done:
			es = self._episode_stats
			print("=== Episode summary ===")
			print(f"Total demands encountered: count={es['demand_count']}, total_capacity={es['demand_capacity']}")
			print(f"Served: count={es['served_count']}, served_capacity={es['served_capacity']}")
			if es.get('expired_count', 0) > 0:
				print(f"Expired (timed-out): count={es.get('expired_count',0)}, capacity={es.get('expired_capacity',0.0)}, penalty={es.get('expired_penalty',0.0)}")
			remaining_count = len(self._state.demands)
			remaining_capacity = sum(float(d.c) for d in self._state.demands)
			print(f"Remaining unserved: count={remaining_count}, capacity={remaining_capacity}")
			for i, td in enumerate(es['agent_total_distances']):
				print(f"Agent {i} total distance: {td}")
			print(f"Total distance this episode: {es['total_distance']}")
			print(f"Episode cumulative reward: {es['episode_reward']}")
			# also return stats in info for external use
			info['episode_stats'] = {
				'counts': {'encountered': es['demand_count'], 'served': es['served_count'], 'remaining': remaining_count},
				'capacities': {'encountered': es['demand_capacity'], 'served': es['served_capacity'], 'remaining': remaining_capacity},
				'served_details': list(es['served_details']),
				'agent_total_distances': list(es['agent_total_distances']),
				'total_distance': es['total_distance'],
				'episode_reward': es['episode_reward'],
			}
		return self._obs(), reward, done, info

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
