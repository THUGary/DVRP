from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PairwiseGraphCritic(nn.Module):
	"""Shared-encoder critic over agent pairs plus optional demand context.

	Inputs:
	    agents: [B, A, F] tensor (e.g., x, y, s, t for each agent).
	    global_ctx: [B, G] tensor summarizing demand/depot graph (optional).

	Outputs:
	    value: [B] tensor with a scalar state-value per batch element.
	"""

	def __init__(self, agent_dim: int, hidden_dim: int = 128, global_dim: int = 0) -> None:
		super().__init__()
		self.agent_dim = int(agent_dim)
		self.hidden_dim = int(hidden_dim)
		self.global_dim = int(global_dim)

		self.agent_mlp = nn.Sequential(
			nn.Linear(self.agent_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
		)

		self.pair_mlp = nn.Sequential(
			nn.Linear(2 * self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
		)

		out_input_dim = self.hidden_dim + self.global_dim
		self.out_mlp = nn.Sequential(
			nn.Linear(out_input_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1),
		)

	def forward(self, agents: torch.Tensor, global_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
		"""Compute state value from agent features and optional global context.

		Args:
		    agents: [B, A, F] float tensor.
		    global_ctx: [B, G] float tensor; if None or mismatched, ignored.
		"""
		if agents.dim() != 3:
			raise ValueError(f"PairwiseGraphCritic expects agents of shape [B,A,F], got {agents.shape}")

		B, A, F = agents.shape
		if A <= 1:
			# Degenerate single-agent case: fall back to mean-pooled embedding.
			h = self.agent_mlp(agents)  # [B, A, H]
			g = h.mean(dim=1)  # [B,H]
		else:
			agents_flat = agents.view(B * A, F)
			h = self.agent_mlp(agents_flat).view(B, A, self.hidden_dim)  # [B,A,H]

			# Build pairwise representations h_i || h_j for all ordered pairs (i,j).
			hi = h.unsqueeze(2).expand(B, A, A, self.hidden_dim)  # [B,A,A,H]
			hj = h.unsqueeze(1).expand(B, A, A, self.hidden_dim)  # [B,A,A,H]
			pair = torch.cat([hi, hj], dim=-1)  # [B,A,A,2H]

			pair_h = self.pair_mlp(pair)  # [B,A,A,H]
			# Aggregate all pairwise relations into a global descriptor.
			g = pair_h.mean(dim=(1, 2))  # [B,H]

		# Fuse global demand/depot context if provided and dimensionally compatible.
		if global_ctx is not None and global_ctx.dim() == 2 and global_ctx.size(0) == B and self.global_dim > 0:
			if global_ctx.size(1) != self.global_dim:
				raise ValueError(f"PairwiseGraphCritic expected global_ctx dim {self.global_dim}, got {global_ctx.size(1)}")
			g = torch.cat([g, global_ctx], dim=-1)  # [B, H+G]

		v = self.out_mlp(g).squeeze(-1)  # [B]
		return v
