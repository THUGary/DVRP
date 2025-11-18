from __future__ import annotations

from typing import Dict, Any, List

import torch

from .base import RLAlgorithm, DecisionRecord


class ReinforceAlgorithm(RLAlgorithm):
	"""Vanilla REINFORCE with exponential moving-average baseline."""

	requires_full_state = False

	def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, args: Any) -> None:
		super().__init__(model, optimizer, device, args)
		self.baseline: float | None = None
		self.logprob_traj: List[torch.Tensor] = []

	def begin_episode(self, episode_idx: int) -> None:
		self.logprob_traj = []

	def record_decision(self, record: DecisionRecord) -> None:
		self.logprob_traj.append(record.log_prob_sum)

	def end_episode(
		self,
		total_reward: float,
		rewards: List[float],
		dones: List[bool],
		env_stats: Dict[str, Any],
	) -> Dict[str, float]:
		if self.baseline is None:
			self.baseline = total_reward
		advantage = total_reward - self.baseline
		self.baseline = 0.9 * self.baseline + 0.1 * total_reward

		if not self.logprob_traj:
			return {"loss": 0.0, "advantage": float(advantage)}

		sum_logprob = torch.stack(self.logprob_traj).sum()
		if not sum_logprob.requires_grad:
			return {"loss": 0.0, "advantage": float(advantage)}

		loss = -advantage * sum_logprob
		if self.optimizer is None:
			raise RuntimeError("REINFORCE requires an optimizer for the policy model.")

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
		self.optimizer.step()

		return {"loss": float(loss.item()), "advantage": float(advantage)}
