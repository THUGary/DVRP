from __future__ import annotations

from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.planner_model.model import DVRPNet

from .base import RLAlgorithm, DecisionRecord
from .sampling import aggregate_state_embedding


class ValueCritic(nn.Module):
	"""Small MLP critic operating on depot + mean-node embeddings."""

	def __init__(self, d_model: int) -> None:
		super().__init__()
		hidden = max(128, d_model)
		self.input_dim = d_model * 2
		self.net = nn.Sequential(
			nn.Linear(self.input_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x).squeeze(-1)


def compute_returns(rewards: List[float], dones: List[bool], gamma: float, device: torch.device) -> torch.Tensor:
	R = 0.0
	returns: List[float] = []
	for reward, done in zip(reversed(rewards), reversed(dones)):
		if done:
			R = 0.0
		R = reward + gamma * R
		returns.append(R)
	returns.reverse()
	return torch.tensor(returns, dtype=torch.float32, device=device)


def evaluate_sample(
	model: DVRPNet,
	critic: ValueCritic,
	sample: Dict[str, Any],
	lateness_lambda: float,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	feats = {k: v.to(device) for k, v in sample["feats"].items()}
	agents = sample["agents"].to(device)
	actions = sample["actions"].to(device)
	history_positions = sample.get("history_positions")
	history_indices = sample.get("history_indices")
	if history_positions is not None:
		history_positions = history_positions.to(device)
	if history_indices is not None:
		history_indices = history_indices.to(device)

	if feats["nodes"].size(1) == 0:
		B = actions.size(0)
		log_prob = torch.zeros(B, device=device)
		entropy = torch.zeros(B, device=device)
		value = critic(torch.zeros(B, critic.input_dim, device=device))
		return log_prob, entropy, value

	enc_nodes, enc_depot, node_mask = model.encoder(feats)
	enc_agents = model.encoder.encode_agents(agents)
	logits = model.decode(
		enc_nodes=enc_nodes,
		enc_depot=enc_depot,
		node_mask=node_mask,
		enc_agents=enc_agents,
		agents_tensor=agents,
		nodes=feats.get("nodes"),
		lateness_lambda=lateness_lambda,
		history_indices=history_indices,
		history_positions=history_positions,
	)
	probs = torch.softmax(logits, dim=-1)
	logp = torch.log_softmax(logits, dim=-1)
	B, A, _ = probs.shape

	log_terms = []
	ent_terms = []
	for b in range(B):
		lp = []
		ent = []
		for a in range(A):
			act = actions[b, a]
			lp.append(logp[b, a, act])
			ent.append((-(probs[b, a] * logp[b, a]).sum()))
		log_terms.append(torch.stack(lp).sum())
		ent_terms.append(torch.stack(ent).sum())

	state_embed = aggregate_state_embedding(enc_nodes, enc_depot, node_mask)
	value = critic(state_embed)
	return torch.stack(log_terms), torch.stack(ent_terms), value.squeeze(-1)


def ppo_update(
	model: DVRPNet,
	critic: ValueCritic,
	opt_policy: torch.optim.Optimizer,
	opt_value: torch.optim.Optimizer,
	decision_steps: List[Dict[str, Any]],
	returns_all: torch.Tensor,
	args: Any,
	device: torch.device,
	lateness_lambda: float,
) -> Dict[str, float]:
	if not decision_steps:
		return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

	indices = torch.tensor([step["step_index"] for step in decision_steps], dtype=torch.long, device=device)
	targets = returns_all[indices]
	old_log_probs = torch.tensor([step["old_log_prob"] for step in decision_steps], dtype=torch.float32, device=device)
	old_values = torch.tensor([step["value"] for step in decision_steps], dtype=torch.float32, device=device)
	advantages = targets - old_values
	if args.normalize_adv and advantages.numel() > 1:
		advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

	total_batches = 0
	accum_policy = 0.0
	accum_value = 0.0
	accum_entropy = 0.0
	batch_size = max(1, min(args.ppo_batch_size, len(decision_steps)))

	for _ in range(args.ppo_epochs):
		perm = torch.randperm(len(decision_steps), device=device)
		for start in range(0, len(decision_steps), batch_size):
			idx = perm[start:start + batch_size]
			batch_samples = [decision_steps[int(i.item())] for i in idx]

			new_log_list = []
			entropy_list = []
			value_list = []
			for sample in batch_samples:
				log_prob, entropy, value = evaluate_sample(
					model,
					critic,
					sample,
					lateness_lambda,
					device,
				)
				new_log_list.append(log_prob.squeeze())
				entropy_list.append(entropy.squeeze())
				value_list.append(value.squeeze())

			new_log_probs = torch.stack(new_log_list)
			entropies = torch.stack(entropy_list)
			values = torch.stack(value_list)

			old_batch = old_log_probs[idx]
			adv_batch = advantages[idx]
			target_batch = targets[idx]

			ratio = torch.exp(new_log_probs - old_batch)
			surr1 = ratio * adv_batch
			surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * adv_batch
			policy_loss = -torch.min(surr1, surr2).mean()
			value_loss = F.mse_loss(values, target_batch)
			entropy_bonus = entropies.mean()

			total_loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus

			opt_policy.zero_grad()
			opt_value.zero_grad()
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
			opt_policy.step()
			opt_value.step()

			accum_policy += policy_loss.item()
			accum_value += value_loss.item()
			accum_entropy += entropy_bonus.item()
			total_batches += 1

	if total_batches == 0:
		return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

	return {
		"policy_loss": accum_policy / total_batches,
		"value_loss": accum_value / total_batches,
		"entropy": accum_entropy / total_batches,
	}


class PPOAlgorithm(RLAlgorithm):
	requires_full_state = True

	def __init__(self, model: DVRPNet, optimizer: torch.optim.Optimizer, device: torch.device, args: Any) -> None:
		super().__init__(model, optimizer, device, args)
		self.critic = ValueCritic(model.d_model).to(device)
		self.critic.train()
		self.value_opt = torch.optim.AdamW(self.critic.parameters(), lr=args.value_lr)
		self.gamma = args.gamma
		self.decision_steps: List[Dict[str, Any]] = []

	def begin_episode(self, episode_idx: int) -> None:
		self.decision_steps = []

	def record_decision(self, record: DecisionRecord) -> None:
		if record.feats is None or record.agents is None or record.actions is None:
			raise ValueError("PPOAlgorithm requires full state tensors; set requires_full_state=True when capturing data.")
		self.decision_steps.append({
			"step_index": record.step_index,
			"feats": record.feats,
			"agents": record.agents,
			"actions": record.actions,
			"old_log_prob": float(record.log_prob_sum.detach().cpu().item()),
			"value": float(record.state_value.detach().cpu().item()) if record.state_value is not None else 0.0,
			"history_positions": record.history_positions,
			"history_indices": record.history_indices,
		})

	def end_episode(
		self,
		total_reward: float,
		rewards: List[float],
		dones: List[bool],
		env_stats: Dict[str, Any],
	) -> Dict[str, float]:
		if not rewards:
			rewards = [0.0]
			dones = [True]
		returns_all = compute_returns(rewards, dones, self.gamma, self.device)
		stats = ppo_update(
			self.model,
			self.critic,
			self.optimizer,
			self.value_opt,
			self.decision_steps,
			returns_all,
			self.args,
			self.device,
			self.args.lateness_lambda,
		)
		return stats
