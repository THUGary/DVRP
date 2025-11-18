from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
from torch.distributions import Categorical

from models.planner_model.model import DVRPNet


def aggregate_state_embedding(enc_nodes: torch.Tensor, enc_depot: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
	"""Produce a compact depot + mean-node embedding used by critics."""
	depot_embed = enc_depot.squeeze(1)
	if enc_nodes.size(1) == 0:
		node_mean = torch.zeros_like(depot_embed)
	else:
		valid = (~node_mask).unsqueeze(-1).float()
		denom = valid.sum(dim=1).clamp(min=1.0)
		node_mean = (enc_nodes * valid).sum(dim=1) / denom
	return torch.cat([depot_embed, node_mean], dim=-1)


def detach_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	"""Detach feature tensors to CPU for later reuse (e.g., PPO)."""
	return {k: v.detach().cpu().clone() for k, v in feats.items()}


def select_targets_with_sampling(
	model: DVRPNet,
	feats: Dict[str, torch.Tensor],
	agents_tensor: torch.Tensor,
	lateness_lambda: float,
	critic: Optional[torch.nn.Module] = None,
	history_positions: Optional[torch.Tensor] = None,
	history_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
	"""Sample target indices per agent using the current policy logits."""
	dev = agents_tensor.device
	if feats["nodes"].size(1) == 0:
		B = agents_tensor.size(0)
		A = agents_tensor.size(1)
		depot_xy = feats["depot"][..., :2].long().squeeze(1)
		sel = torch.zeros(B, A, dtype=torch.long, device=dev)
		dest_xy = depot_xy.unsqueeze(1).repeat(1, A, 1)
		log_probs = torch.zeros(B, A, dtype=torch.float32, device=dev)
		value = None
		if critic is not None:
			embed = torch.zeros(B, critic.input_dim, device=dev)
			value = critic(embed)
		return sel, dest_xy, log_probs, value

	enc_nodes, enc_depot, node_mask = model.encoder(feats)
	enc_agents = model.encoder.encode_agents(agents_tensor)
	logits = model.decode(
		enc_nodes=enc_nodes,
		enc_depot=enc_depot,
		node_mask=node_mask,
		enc_agents=enc_agents,
		agents_tensor=agents_tensor,
		nodes=feats["nodes"],
		lateness_lambda=lateness_lambda,
		history_indices=history_indices,
		history_positions=history_positions,
	)

	probs = torch.softmax(logits, dim=-1)
	logp = torch.log_softmax(logits, dim=-1)
	B, A, _ = probs.shape
	N = feats["nodes"].size(1)
	depot_xy = feats["depot"][..., :2].long().squeeze(1)
	node_xy = feats["nodes"][..., :2].long()

	sel = torch.zeros(B, A, dtype=torch.long, device=dev)
	dest_xy = torch.zeros(B, A, 2, dtype=torch.long, device=dev)
	log_probs = torch.zeros(B, A, dtype=torch.float32, device=dev)

	for b in range(B):
		for a in range(A):
			cat = Categorical(probs[b, a])
			idx = cat.sample()
			sel[b, a] = idx
			log_probs[b, a] = logp[b, a, idx]
			if 1 <= idx <= N:
				dest_xy[b, a] = node_xy[b, idx - 1]
			else:
				dest_xy[b, a] = depot_xy[b]

	value = None
	if critic is not None:
		# Critic may operate on aggregated embeddings or directly on agents graph.
		# Here we pass agents_tensor so graph-based critics can use pairwise relations.
		try:
			value = critic(agents_tensor)
		except TypeError:
			# Fallback for old critics expecting aggregated state embedding
			state_embed = aggregate_state_embedding(enc_nodes, enc_depot, node_mask)
			value = critic(state_embed)

	return sel, dest_xy, log_probs, value
