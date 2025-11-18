from __future__ import annotations

from typing import Dict, Any, List, Tuple

import torch

from .base import RLAlgorithm, DecisionRecord
from .critics import PairwiseGraphCritic


class ReinforceAlgorithm(RLAlgorithm):
	"""REINFORCE with exponential moving-average baseline and optional graph critic.

	The critic is not used to update the policy (still Monte-Carlo), but it
	can be logged or later extended for variance reduction / actor-critic.
	"""

	requires_full_state = True

	def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, args: Any) -> None:
		super().__init__(model, optimizer, device, args)
		self.baseline: float | None = None
		self.logprob_traj: List[torch.Tensor] = []
		self.critic = PairwiseGraphCritic(agent_dim=4, hidden_dim=max(128, getattr(model, "d_model", 128))).to(device)
		self.critic.eval()  # currently unused for training
		self._tmp_snapshots: List[Tuple[torch.Tensor, float]] = []

	def begin_episode(self, episode_idx: int) -> None:
		self.logprob_traj = []
		self._tmp_snapshots = []

	def record_decision(self, record: DecisionRecord) -> None:
		self.logprob_traj.append(record.log_prob_sum)
		if record.agents is not None:
			time_val = float(record.current_time) if record.current_time is not None else 0.0
			self._tmp_snapshots.append((record.agents, time_val))

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

		# Optional: compute critic estimates for logging (no grad)
		critic_value = 0.0
		if self._tmp_snapshots:
			with torch.no_grad():
				agents_aug = []
				for snap_agents, snap_time in self._tmp_snapshots:
					time_tensor = torch.full(
						(snap_agents.size(0), snap_agents.size(1), 1),
						float(snap_time),
						dtype=snap_agents.dtype,
					)
					agents_aug.append(torch.cat([snap_agents, time_tensor], dim=-1))
				agents_cat = torch.stack(agents_aug, dim=0).to(self.device)
				agents_flat = agents_cat.view(-1, agents_cat.size(-2), agents_cat.size(-1))
				v = self.critic(agents_flat)
				critic_value = float(v.mean().cpu().item())

		return {"loss": float(loss.item()), "advantage": float(advantage), "critic_value": critic_value}
