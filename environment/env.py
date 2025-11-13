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
		expiry_penalty_scale: float = 5.0,
		switch_penalty_scale: float = 0.01,
		capacity_reward_scale: float = 10.0,
		exploration_history_n: int = 0,
		exploration_penalty_scale: float = 0.0,
	) -> None:
		self.width = width
		self.height = height
		self.max_time = max_time
		self.depot = depot
		self.capacity = capacity
		self._generator = generator
		self.expiry_penalty_scale = expiry_penalty_scale
		self.switch_penalty_scale = switch_penalty_scale
		self.capacity_reward_scale = capacity_reward_scale
		self.exploration_history_n = max(0, int(exploration_history_n))
		self.exploration_penalty_scale = float(exploration_penalty_scale)
		self._state: Optional[EnvState] = None

		# cache for resolved full capacity to avoid repeated imports
		self._resolved_full_capacity: Optional[int] = None
		self._prev_actions: List[Action] = []

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
		num_agents = getattr(self, "_num_agents", None)
		if num_agents is None or num_agents <= 0:
			num_agents = 1
			self._num_agents = num_agents
		agent_states = self._spawn_initial_agent_states(num_agents, seed)
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
		self._episode_stats["switch_count"] = 0
		self._episode_stats["switch_penalty"] = 0.0
		self._prev_actions = [(0, 0) for _ in agent_states]
		# maintain per-agent position history for exploration penalty
		self._pos_history: List[List[Tuple[int,int]]] = [[(a.x, a.y)] for a in agent_states]
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
		prev_states = [AgentState(x=a.x, y=a.y, s=a.s) for a in self._state.agent_states]
		prev_positions = [(a.x, a.y) for a in prev_states]
		# stability penalty: count agents that changed their intended direction
		switch_count = 0
		if self._prev_actions:
			for idx, act in enumerate(actions):
				if idx < len(self._prev_actions):
					prev_act = self._prev_actions[idx]
					if prev_act != (0, 0) and act != prev_act:
						switch_count += 1
		else:
			self._prev_actions = [(0, 0) for _ in actions]
		capacity_reward = 0.0
		candidate_states: List[AgentState] = []
		for i, (dx, dy) in enumerate(actions):
			a_prev = prev_states[i]
			nx = max(0, min(self.width - 1, a_prev.x + max(-1, min(1, dx))))
			ny = max(0, min(self.height - 1, a_prev.y + max(-1, min(1, dy))))
			# if agent arrives at depot, refill to full capacity
			new_s = a_prev.s
			if (nx, ny) == self.depot:
				new_s = self._full_capacity()
			candidate_states.append(AgentState(x=nx, y=ny, s=new_s))

		# resolve collisions with "first-wins" policy:
		# If multiple agents attempt to occupy the same non-depot cell this step, the lowest-index agent keeps the move
		# (winner), and all other agents (losers) revert to their previous state.
		pos_to_indices: Dict[Tuple[int, int], List[int]] = {}
		for idx, st in enumerate(candidate_states):
			pos_to_indices.setdefault((st.x, st.y), []).append(idx)
		collided_agents: List[int] = []
		for pos, indices in pos_to_indices.items():
			if len(indices) > 1 and pos != self.depot:
				indices_sorted = sorted(indices)
				winner = indices_sorted[0]
				losers = indices_sorted[1:]
				for idx in losers:
					prev = prev_states[idx]
					candidate_states[idx] = AgentState(x=prev.x, y=prev.y, s=prev.s)
				collided_agents.extend(losers)

	# second pass to ensure uniqueness after reverting (should hold if prev states were unique)
		self._state.agent_states = candidate_states
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

		# --- Exploration revisit penalty ---
		exploration_penalty = 0.0
		if self.exploration_history_n > 1:
			for idx, a in enumerate(self._state.agent_states):
				# ensure history list exists
				if idx >= len(self._pos_history):
					self._pos_history.append([])
				hist = self._pos_history[idx]
				cur_pos = (a.x, a.y)
				# look back positions at t-2 .. t-n (skip immediate previous which is at end)
				# hist stores chronological positions; last element is previous step position
				look_back = min(len(hist)-1, self.exploration_history_n)
				if look_back >= 2:
					# indices: -2 (t-1), -3 (t-2), ... but we need t-2 .. t-n relative to current t after move
					# Because we append later, hist[-1] is previous position (t-1). We compare current position with older ones.
					for offset, h_index in enumerate(range(2, look_back+1), start=1):
						past_pos = hist[-h_index]
						if past_pos == cur_pos:
							# weight rule: if match with position from t-2 (closest older), penalty = (exploration_history_n-1);
							# if match with t-n (farthest), penalty = 1. offset=1 corresponds to t-2.
							weight = max(1, (self.exploration_history_n - offset))
							exploration_penalty += float(weight)
		# scale exploration penalty (negative contribution)
		exploration_penalty_value = - self.exploration_penalty_scale * exploration_penalty if exploration_penalty > 0 else 0.0

		# --- Reward aggregation ---
		capacity_reward_term = float(self.capacity_reward_scale) * float(capacity_reward)
		switch_penalty_term = - float(self.switch_penalty_scale) * float(switch_count)
		reward = capacity_reward_term + expiry_penalty 		# + switch_penalty_term  + exploration_penalty_value

		# update episode-level stats
		self._episode_stats["served_count"] += served_count
		self._episode_stats["served_capacity"] += served_capacity
		self._episode_stats.setdefault('capacity_reward_term', 0)
		self._episode_stats["capacity_reward_term"] += capacity_reward_term
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
		self._episode_stats["switch_count"] += switch_count
		self._episode_stats["switch_penalty"] += float(switch_penalty_term)
		if self.exploration_history_n > 1:
			self._episode_stats.setdefault("exploration_penalty_raw", 0.0)
			self._episode_stats.setdefault("exploration_penalty_value", 0.0)
			self._episode_stats["exploration_penalty_raw"] += exploration_penalty
			self._episode_stats["exploration_penalty_value"] += exploration_penalty_value
		self._prev_actions = list(actions)
		# update position histories AFTER computing penalty
		for idx, a in enumerate(self._state.agent_states):
			if idx >= len(self._pos_history):
				self._pos_history.append([])
			hist = self._pos_history[idx]
			hist.append((a.x, a.y))
			# trim history to n+2 (keep some buffer) to bound memory
			max_keep = max(2, self.exploration_history_n + 2)
			if len(hist) > max_keep:
				self._pos_history[idx] = hist[-max_keep:]

		# 4) time update
		self._state.time += 1
		done = self._state.time >= self.max_time
		# on episode end, print aggregated statistics and return them in info
		info: Dict = {}
		if collided_agents:
			info["collisions"] = len(collided_agents)
			self._episode_stats.setdefault("collision_count", 0)
			self._episode_stats["collision_count"] += len(collided_agents)
		if done:
			es = self._episode_stats
			print("=== Episode summary ===")
			print(f"Total demands encountered: count={es['demand_count']}, total_capacity={es['demand_capacity']}")
			print(f"Served: count={es['served_count']}, served_capacity={es['served_capacity']}, capacity_reward={es.get('capacity_reward_term', 0.0)}")
			if es.get('expired_count', 0) > 0:
				print(f"Expired (timed-out): count={es.get('expired_count',0)}, capacity={es.get('expired_capacity',0.0)}, penalty={es.get('expired_penalty',0.0)}")
			if es.get('collision_count', 0) > 0:
				print(f"Agent collision resolutions: {es.get('collision_count', 0)}")
			if es.get('switch_count', 0) > 0:
				print(f"Target switches penalized: count={es.get('switch_count', 0)}, penalty={es.get('switch_penalty', 0.0)}")
			if self.exploration_history_n > 1:
				print(f"Exploration revisit penalty: raw={es.get('exploration_penalty_raw', 0.0)}, value={es.get('exploration_penalty_value', 0.0)}")
			remaining_count = len(self._state.demands)
			remaining_capacity = sum(float(d.c) for d in self._state.demands)
			# print(f"Remaining unserved: count={remaining_count}, capacity={remaining_capacity}")
			# for i, td in enumerate(es['agent_total_distances']):
			# 	print(f"Agent {i} total distance: {td}")
			# print(f"Total distance this episode: {es['total_distance']}")
			print(f"Episode cumulative reward: {es['episode_reward']}")
			# also return stats in info for external use
			info['episode_stats'] = {
				'counts': {'encountered': es['demand_count'], 'served': es['served_count'], 'remaining': remaining_count},
				'capacities': {'encountered': es['demand_capacity'], 'served': es['served_capacity'], 'remaining': remaining_capacity},
				'served_details': list(es['served_details']),
				'agent_total_distances': list(es['agent_total_distances']),
				'total_distance': es['total_distance'],
				'episode_reward': es['episode_reward'],
				'switch_penalty': es.get('switch_penalty', 0.0),
				'switch_count': es.get('switch_count', 0),
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

	def _spawn_initial_agent_states(self, num_agents: int, seed: Optional[int]) -> List[AgentState]:
		if num_agents <= 0:
			return []
		dx, dy = self.depot
		full = self._full_capacity()
		return [AgentState(x=dx, y=dy, s=full) for _ in range(num_agents)]
