from __future__ import annotations

from typing import Any, Dict, List

import torch

from .base import RLAlgorithm, DecisionRecord


class POMOAlgorithm(RLAlgorithm):
    """Policy Optimization with Multiple Optima (POMO) for DVRPNet.

    Each environment instance is replayed multiple times with different policy
    stochasticity. The competing rollouts form a self-referenced baseline that
    stabilizes the REINFORCE-style gradient estimate.
    """

    requires_full_state = False

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, args: Any) -> None:
        super().__init__(model, optimizer, device, args)
        self.entropy_coef = float(getattr(args, "pomo_entropy_coef", 0.0))
        self.grad_clip = float(getattr(args, "pomo_grad_clip", 1.0))
        baseline_choice = str(getattr(args, "pomo_baseline", "leave_one_out")).lower()
        self.leave_one_out = baseline_choice != "mean"
        self._step_logprobs: List[torch.Tensor] = []
        self._step_entropies: List[torch.Tensor] = []
        self._traj_logprobs: List[torch.Tensor] = []
        self._traj_entropies: List[torch.Tensor] = []
        self._traj_rewards: List[float] = []

    def begin_group(self, group_idx: int) -> None:  # noqa: D401 - see base class
        self._traj_logprobs = []
        self._traj_entropies = []
        self._traj_rewards = []

    def begin_episode(self, episode_idx: int) -> None:  # noqa: D401 - see base class
        self._step_logprobs = []
        self._step_entropies = []

    def record_decision(self, record: DecisionRecord) -> None:
        self._step_logprobs.append(record.log_prob_sum)
        if record.entropy_sum is not None:
            self._step_entropies.append(record.entropy_sum)

    def end_episode(
        self,
        total_reward: float,
        rewards: List[float],
        dones: List[bool],
        env_stats: Dict[str, Any],
    ) -> Dict[str, float]:
        if self._step_logprobs:
            traj_logprob = torch.stack(self._step_logprobs).sum()
        else:
            traj_logprob = torch.zeros((), device=self.device, dtype=torch.float32)
        if self._step_entropies:
            traj_entropy = torch.stack(self._step_entropies).sum()
        else:
            traj_entropy = torch.zeros((), device=self.device, dtype=torch.float32)

        self._traj_logprobs.append(traj_logprob)
        self._traj_entropies.append(traj_entropy)
        self._traj_rewards.append(float(total_reward))
        # Per-trajectory stats are still useful for debugging
        return {
            "traj_reward": float(total_reward),
            "traj_steps": float(len(rewards)),
        }

    def end_group(self) -> Dict[str, float]:
        if not self._traj_rewards:
            return {}
        rewards = torch.tensor(self._traj_rewards, device=self.device, dtype=torch.float32)
        traj_log = torch.stack(self._traj_logprobs)
        traj_entropy = torch.stack(self._traj_entropies)
        n = rewards.numel()

        if n == 1 or not self.leave_one_out:
            baseline = rewards.mean()
            advantages = rewards - baseline
        else:
            sum_rewards = rewards.sum()
            baseline = (sum_rewards - rewards) / (n - 1)
            advantages = rewards - baseline

        loss = -(advantages.detach() * traj_log).mean()
        if self.entropy_coef > 0.0:
            loss -= self.entropy_coef * traj_entropy.mean()

        if self.optimizer is None:
            raise RuntimeError("POMOAlgorithm requires an optimizer.")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        reward_mean = float(rewards.mean().item())
        reward_std = float(rewards.std(unbiased=False).item()) if n > 1 else 0.0
        stats = {
            "loss": float(loss.item()),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_max": float(rewards.max().item()),
            "reward_min": float(rewards.min().item()),
        }
        return stats
