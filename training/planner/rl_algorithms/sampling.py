from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
from torch.distributions import Categorical

from models.planner_model.model import DVRPNet, _broadcast_time_like


def aggregate_state_embedding(
	enc_nodes: torch.Tensor,
	enc_depot: torch.Tensor,
	node_mask: torch.Tensor,
	current_time: Optional[torch.Tensor] = None,
) -> torch.Tensor:
	"""Produce a compact depot + mean-node embedding used by critics."""
	depot_embed = enc_depot.squeeze(1)
	if enc_nodes.size(1) == 0:
		node_mean = torch.zeros_like(depot_embed)
	else:
		valid = (~node_mask).unsqueeze(-1).float()
		denom = valid.sum(dim=1).clamp(min=1.0)
		node_mean = (enc_nodes * valid).sum(dim=1) / denom
	ctx = torch.cat([depot_embed, node_mean], dim=-1)
	if current_time is not None:
		t_scalar = current_time.mean(dim=1, keepdim=True)
		ctx = torch.cat([ctx, t_scalar], dim=-1)
	return ctx


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
	target_queue_len: int = 1,
 	current_time: torch.Tensor | float | int = 0,
 	agent_times: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
	"""Sample target indices per agent using the current policy logits."""
	dev = agents_tensor.device
	queue_len = max(1, int(target_queue_len))
	B = agents_tensor.size(0)
	A = agents_tensor.size(1)
	time_tensor = _broadcast_time_like(
		agent_times if agent_times is not None else current_time,
		B,
		A,
		dev,
		agents_tensor.dtype,
	)
	if feats["nodes"].size(1) == 0:
		depot_xy = feats["depot"].long().squeeze(1)
		sel = torch.zeros(B, A, dtype=torch.long, device=dev)
		dest_xy = depot_xy.unsqueeze(1).repeat(1, A, 1)
		log_probs = torch.zeros(B, A, dtype=torch.float32, device=dev)
		value = None
		if critic is not None:
			agents_for_critic = torch.cat([agents_tensor, time_tensor.unsqueeze(-1)], dim=-1)
			try:
				value = critic(agents_for_critic)
			except Exception:
				value = torch.zeros(B, device=dev)
		queue_indices = torch.zeros(B, A, queue_len, dtype=torch.long, device=dev)
		queue_coords = torch.zeros(B, A, queue_len, 2, dtype=torch.long, device=dev)
		queue_indices[..., 0] = sel
		queue_coords[..., 0, :] = dest_xy
		return sel, dest_xy, log_probs, value, queue_indices, queue_coords

	enc_nodes, enc_depot, node_mask = model.encoder(feats)
	enc_agents = model.encoder.encode_agents(agents_tensor)
	logits = model.decode(
		enc_nodes=enc_nodes,
		enc_depot=enc_depot,
		node_mask=node_mask,
		enc_agents=enc_agents,
		agents_tensor=agents_tensor,
		agent_times=time_tensor,
		nodes=feats["nodes"],
		lateness_lambda=lateness_lambda,
		history_indices=history_indices,
		history_positions=history_positions,
	)

	probs = torch.softmax(logits, dim=-1)
	logp = torch.log_softmax(logits, dim=-1)
	B, A, _ = probs.shape
	N = feats["nodes"].size(1)
	depot_xy = feats["depot"].long().squeeze(1)
	node_xy = feats["nodes"][..., :2].long()
	queue_indices = torch.zeros(B, A, queue_len, dtype=torch.long, device=dev)
	queue_coords = torch.zeros(B, A, queue_len, 2, dtype=torch.long, device=dev)

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

	queue_indices[..., 0] = sel
	queue_coords[..., 0, :] = dest_xy

	value = None
	if critic is not None:
		# Build global context from demand/depot encoder outputs
		global_ctx = aggregate_state_embedding(enc_nodes, enc_depot, node_mask, time_tensor)
		try:
			agents_for_critic = torch.cat([agents_tensor, time_tensor.unsqueeze(-1)], dim=-1)
			value = critic(agents_for_critic, global_ctx)  # type: ignore[arg-type]
		except TypeError:
			# Backward-compatible fallback: critic may only accept agents or state embedding
			try:
				agents_for_critic = torch.cat([agents_tensor, time_tensor.unsqueeze(-1)], dim=-1)
				value = critic(agents_for_critic)
			except TypeError:
				value = critic(global_ctx)

	# --- Autoregressive queue generation (greedy for steps > 1) ---
	if queue_len > 1 and feats["nodes"].size(1) > 0:
		mask_roll = node_mask.clone()
		agents_roll = agents_tensor.clone()
		agent_times_roll = time_tensor.clone()
		nodes = feats["nodes"]

		def mark_selected(mask: torch.Tensor, indices: torch.Tensor) -> None:
			for b in range(indices.size(0)):
				for a in range(indices.size(1)):
					idx = int(indices[b, a].item())
					if 1 <= idx <= nodes.size(1):
						mask[b, idx - 1] = True

		def advance_agents(state: torch.Tensor, coords: torch.Tensor, indices: torch.Tensor, times: torch.Tensor) -> None:
			if nodes.size(1) == 0:
				return
			for b in range(indices.size(0)):
				for a in range(indices.size(1)):
					idx = int(indices[b, a].item())
					dst = coords[b, a].to(state.dtype)
					cur = state[b, a, :2]
					dist = (cur[0] - dst[0]).abs() + (cur[1] - dst[1]).abs()
					state[b, a, 0:2] = dst
					times[b, a] = times[b, a] + dist
					if 1 <= idx <= nodes.size(1):
						demand = nodes[b, idx - 1, 3].to(state.dtype)
						state[b, a, 2] = torch.clamp(state[b, a, 2] - demand, min=0.0)

		mark_selected(mask_roll, sel)
		advance_agents(agents_roll, dest_xy, sel, agent_times_roll)

		for step in range(1, queue_len):
			if (~mask_roll).sum().item() == 0:
				break
			enc_agents_step = model.encoder.encode_agents(agents_roll)
			logits_step = model.decode(
				enc_nodes=enc_nodes,
				enc_depot=enc_depot,
				node_mask=mask_roll,
				enc_agents=enc_agents_step,
				agents_tensor=agents_roll,
				agent_times=agent_times_roll,
				nodes=feats["nodes"],
				lateness_lambda=lateness_lambda,
				history_indices=history_indices,
				history_positions=history_positions,
			)
			next_idx = torch.argmax(logits_step, dim=-1)
			queue_indices[..., step] = next_idx
			for b in range(B):
				for a in range(A):
					idx = int(next_idx[b, a].item())
					if 1 <= idx <= N:
						queue_coords[b, a, step] = node_xy[b, idx - 1]
					else:
						queue_coords[b, a, step] = depot_xy[b]
			mark_selected(mask_roll, next_idx)
			advance_agents(agents_roll, queue_coords[..., step, :], next_idx, agent_times_roll)

	return sel, dest_xy, log_probs, value, queue_indices, queue_coords
