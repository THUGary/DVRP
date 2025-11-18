from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PairwiseGraphCritic(nn.Module):
	"""Shared-encoder critic operating on agent pairwise relations.

	Inputs:
	    agents: [B, A, F] tensor (e.g., x, y, s, t for each agent).

	Outputs:
	    value: [B] tensor with a scalar state-value per batch element.
	"""

	def __init__(self, agent_dim: int, hidden_dim: int = 128) -> None:
		super().__init__()
		self.agent_dim = int(agent_dim)
		self.hidden_dim = int(hidden_dim)

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

		self.out_mlp = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1),
		)

	def forward(self, agents: torch.Tensor, global_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
		"""Compute state value from agent features and optional global context.

		Args:
		    agents: [B, A, F] float tensor.
		    global_ctx: currently unused; reserved for future extensions.
		"""
		if agents.dim() != 3:
			raise ValueError(f"PairwiseGraphCritic expects agents of shape [B,A,F], got {agents.shape}")

		B, A, F = agents.shape
		if A <= 1:
			# Degenerate single-agent case: fall back to mean-pooled embedding.
			h = self.agent_mlp(agents)  # [B, A, H]
			g = h.mean(dim=1)
			v = self.out_mlp(g).squeeze(-1)
			return v

		agents_flat = agents.view(B * A, F)
		h = self.agent_mlp(agents_flat).view(B, A, self.hidden_dim)  # [B,A,H]

		# Build pairwise representations h_i || h_j for all ordered pairs (i,j).
		hi = h.unsqueeze(2).expand(B, A, A, self.hidden_dim)  # [B,A,A,H]
		hj = h.unsqueeze(1).expand(B, A, A, self.hidden_dim)  # [B,A,A,H]
		pair = torch.cat([hi, hj], dim=-1)  # [B,A,A,2H]

		pair_h = self.pair_mlp(pair)  # [B,A,A,H]
		# Aggregate all pairwise relations into a global descriptor.
		g = pair_h.mean(dim=(1, 2))  # [B,H]

		v = self.out_mlp(g).squeeze(-1)  # [B]
		return v
