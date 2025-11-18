from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class RolloutStep:
    """Container for a single PPO transition step.

    We store only the tensors needed for PPO update; feats/agents are kept
    as detached CPU copies to save GPU memory and be serializable.
    """

    step_index: int
    feats: Dict[str, torch.Tensor]
    agents: torch.Tensor
    actions: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    current_time: torch.Tensor
    reward: float
    done: bool


class RolloutBuffer:
    """Simple rollout buffer for on-policy PPO with optional GAE.

    This buffer is episode-agnostic; caller is responsible for providing
    per-step rewards and done flags in temporal order.
    """

    def __init__(self) -> None:
        self.steps: List[RolloutStep] = []
        self._device: torch.device | None = None

    @property
    def empty(self) -> bool:
        return len(self.steps) == 0

    def add(
        self,
        *,
        step_index: int,
        feats: Dict[str, torch.Tensor],
        agents: torch.Tensor,
        actions: torch.Tensor,
        log_prob_sum: torch.Tensor,
        value: torch.Tensor,
        current_time: torch.Tensor | float | int,
        reward: float,
        done: bool,
    ) -> None:
        """Append a new step to the buffer.

        Args:
            step_index: integer index within the episode (for reference only).
            feats: dict of encoder input tensors (detached to CPU).
            agents: [B,A,4] tensor (detached to CPU).
            actions: [B,A] long tensor (detached to CPU).
            log_prob_sum: scalar log-prob tensor (will be stored as tensor).
            value: scalar state-value tensor.
            reward: scalar float reward for this env step.
            done: whether this step ended the episode.
        """
        # We keep feats/agents/actions on CPU to avoid GPU memory growth.
        feats_cpu = {k: v.detach().cpu().clone() for k, v in feats.items()}
        agents_cpu = agents.detach().cpu().clone()
        actions_cpu = actions.detach().cpu().clone()
        logp_cpu = log_prob_sum.detach().cpu().clone()
        value_cpu = value.detach().cpu().clone()

        if isinstance(current_time, torch.Tensor):
            time_cpu = current_time.detach().cpu().clone().float()
        else:
            time_cpu = torch.tensor(float(current_time), dtype=torch.float32)
        step = RolloutStep(
            step_index=step_index,
            feats=feats_cpu,
            agents=agents_cpu,
            actions=actions_cpu,
            log_prob=logp_cpu,
            value=value_cpu,
            current_time=time_cpu,
            reward=float(reward),
            done=bool(done),
        )
        self.steps.append(step)

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Stack buffer into batched tensors on the given device.

        Returns a dict containing batched log_probs, values, rewards, dones
        and an index array referencing each step. Feats/agents/actions are
        kept step-wise and consumed via the original list when re-evaluating
        the policy.
        """
        if not self.steps:
            raise RuntimeError("RolloutBuffer is empty; cannot stack tensors.")

        self._device = device
        log_probs = torch.stack([s.log_prob for s in self.steps], dim=0).to(device)  # [T]
        values = torch.stack([s.value for s in self.steps], dim=0).to(device)        # [T]
        rewards = torch.tensor([s.reward for s in self.steps], dtype=torch.float32, device=device)  # [T]
        dones = torch.tensor([s.done for s in self.steps], dtype=torch.bool, device=device)         # [T]
        indices = torch.tensor([s.step_index for s in self.steps], dtype=torch.long, device=device) # [T]

        return {
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "indices": indices,
        }

    def compute_returns_and_advantages(
        self,
        *,
        gamma: float,
        gae_lambda: float = 1.0,
        use_gae: bool = True,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages for all steps in the buffer.

        If use_gae is False, plain Monte Carlo returns are used and
        advantages = returns - values.
        """
        if not self.steps:
            raise RuntimeError("RolloutBuffer is empty; cannot compute returns.")

        dev = device or (self._device or torch.device("cpu"))
        stacked = self.to_tensors(dev)
        rewards = stacked["rewards"]  # [T]
        values = stacked["values"]    # [T]
        dones = stacked["dones"]      # [T]

        T = rewards.size(0)
        returns = torch.zeros(T, dtype=torch.float32, device=dev)
        advantages = torch.zeros(T, dtype=torch.float32, device=dev)

        if use_gae:
            next_value = torch.zeros((), dtype=torch.float32, device=dev)
            gae = torch.zeros((), dtype=torch.float32, device=dev)
            for t in reversed(range(T)):
                done = dones[t]
                mask = 0.0 if done else 1.0
                delta = rewards[t] + gamma * next_value * mask - values[t]
                gae = delta + gamma * gae_lambda * mask * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
                next_value = values[t]
        else:
            R = torch.zeros((), dtype=torch.float32, device=dev)
            for t in reversed(range(T)):
                if dones[t]:
                    R = torch.zeros((), dtype=torch.float32, device=dev)
                R = rewards[t] + gamma * R
                returns[t] = R
            advantages = returns - values

        return returns, advantages

    def clear(self) -> None:
        self.steps.clear()
        self._device = None
