from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

import torch


@dataclass
class DecisionRecord:
	"""Container for data captured at each policy decision step."""

	step_index: int
	log_prob_sum: torch.Tensor
	current_time: float | int | torch.Tensor | None = None
	feats: Optional[Dict[str, torch.Tensor]] = None
	agents: Optional[torch.Tensor] = None
	actions: Optional[torch.Tensor] = None
	state_value: Optional[torch.Tensor] = None
	history_positions: Optional[torch.Tensor] = None
	history_indices: Optional[torch.Tensor] = None
	queue_indices: Optional[torch.Tensor] = None
	queue_coords: Optional[torch.Tensor] = None
	reward: float = 0.0
	done: bool = False


class RLAlgorithm(ABC):
	"""Abstract base class for planner RL algorithms."""

	requires_full_state: bool = False

	def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device, args: Any) -> None:
		self.model = model
		self.optimizer = optimizer
		self.device = device
		self.args = args

	def begin_episode(self, episode_idx: int) -> None:
		"""Hook invoked at the start of every episode."""
		pass

	@abstractmethod
	def record_decision(self, record: DecisionRecord) -> None:
		"""Store required per-step data for later updates."""
		raise NotImplementedError

	@abstractmethod
	def end_episode(
		self,
		total_reward: float,
		rewards: List[float],
		dones: List[bool],
		env_stats: Dict[str, Any],
	) -> Dict[str, float]:
		"""Finalize the episode and run the algorithm-specific update."""
		raise NotImplementedError
