from __future__ import annotations

from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.planner_model.model import DVRPNet

from .base import RLAlgorithm, DecisionRecord
from .sampling import aggregate_state_embedding
from .critics import PairwiseGraphCritic
from .rollout_buffer import RolloutBuffer


def evaluate_sample(
	model: DVRPNet,
	critic: nn.Module,
	sample: Dict[str, Any],
	lateness_lambda: float,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	feats = {k: v.to(device) for k, v in sample["feats"].items()}
	agents = sample["agents"].to(device)
	actions = sample["actions"].to(device)
	history_positions = sample.get("history_positions")
	history_target_coords = sample.get("history_target_coords")
	if history_positions is not None:
		history_positions = history_positions.to(device)
	if history_target_coords is not None:
		history_target_coords = history_target_coords.to(device)

	if feats["nodes"].size(1) == 0:
		B = actions.size(0)
		log_prob = torch.zeros(B, device=device)
		entropy = torch.zeros(B, device=device)
		# fall back to zeros if critic cannot handle empty state
		try:
			value = critic(torch.zeros(B, 1, 4, device=device))  # [B,A,F] degenerate
		except Exception:
			value = torch.zeros(B, device=device)
		return log_prob, entropy, value

	enc_nodes, enc_depot, node_mask = model.encoder(feats)
	logits = model.decode(
		enc_nodes=enc_nodes,
		enc_depot=enc_depot,
		node_mask=node_mask,
		agents_tensor=agents,
		nodes=feats.get("nodes"),
		lateness_lambda=lateness_lambda,
		history_positions=history_positions,
		history_target_coords=history_target_coords,
	)
	probs = torch.softmax(logits, dim=-1)
	logp = torch.log_softmax(logits, dim=-1)
	B, A, _ = probs.shape
	# global context summarizing demand/depot graph for critic
	global_ctx = aggregate_state_embedding(enc_nodes, enc_depot, node_mask)

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

	# critic operates on agent graph; pass agents tensor
	value = critic(agents, global_ctx)  # type: ignore[arg-type]
	return torch.stack(log_terms), torch.stack(ent_terms), value.squeeze(-1)


def ppo_update(
	model: DVRPNet,
	critic: nn.Module,
	opt_policy: torch.optim.Optimizer,
	opt_value: torch.optim.Optimizer,
	buffer: RolloutBuffer,
	args: Any,
	device: torch.device,
	lateness_lambda: float,
	gamma: float,
	gae_lambda: float = 0.95,
	use_gae: bool = True,
) -> Dict[str, float]:
	"""Run a PPO update using transitions from the rollout buffer.

	For now we assume a single contiguous rollout (one episode). Later it can
	be extended to multi-episode rollouts simply by feeding more steps.
	"""
	if buffer.empty:
		return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

	stacked = buffer.to_tensors(device)
	old_log_probs = stacked["log_probs"].view(-1)
	values = stacked["values"].view(-1)
	returns, advantages = buffer.compute_returns_and_advantages(
		gamma=gamma,
		gae_lambda=gae_lambda,
		use_gae=use_gae,
		device=device,
	)
	advantages = advantages.view(-1)
	returns = returns.view(-1)
	if args.normalize_adv and advantages.numel() > 1:
		advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

	def mean_std(t: torch.Tensor) -> tuple[float, float]:
		if t.numel() == 0:
			return 0.0, 0.0
		mean = float(t.mean().item())
		std = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
		return mean, std

	value_mean, value_std = mean_std(values)
	return_mean, return_std = mean_std(returns)
	adv_mean, adv_std = mean_std(advantages)

	total_batches = 0
	accum_policy = 0.0
	accum_value = 0.0
	accum_entropy = 0.0
	ratio_mean_accum = 0.0
	ratio_std_accum = 0.0
	T = len(buffer.steps)
	batch_size = max(1, min(args.ppo_batch_size, T))

	for _ in range(args.ppo_epochs):
		perm = torch.randperm(T, device=device)
		for start in range(0, T, batch_size):
			idx = perm[start:start + batch_size]
			new_log_list = []
			entropy_list = []
			value_list = []
			for i in idx.tolist():
				step = buffer.steps[i]
				log_prob, entropy, value = evaluate_sample(
					model,
					critic,
					{
						"feats": step.feats,
						"agents": step.agents,
						"actions": step.actions,
					},
					lateness_lambda,
					device,
				)
				new_log_list.append(log_prob.squeeze())
				entropy_list.append(entropy.squeeze())
				value_list.append(value.squeeze())

			new_log_probs = torch.stack(new_log_list)
			entropies = torch.stack(entropy_list)
			values_pred = torch.stack(value_list)

			old_batch = old_log_probs[idx]
			adv_batch = advantages[idx]
			target_batch = returns[idx]

			ratio = torch.exp(new_log_probs - old_batch)
			ratio_mean_accum += ratio.mean().item()
			ratio_std_accum += ratio.std(unbiased=False).item()
			surr1 = ratio * adv_batch
			surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * adv_batch
			policy_loss = -torch.min(surr1, surr2).mean()
			value_loss = F.mse_loss(values_pred, target_batch)
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
		return {
			"policy_loss": 0.0,
			"value_loss": 0.0,
			"entropy": 0.0,
			"ratio_mean": 1.0,
			"ratio_std": 0.0,
			"value_pred_mean": value_mean,
			"value_pred_std": value_std,
			"returns_mean": return_mean,
			"returns_std": return_std,
			"adv_mean": adv_mean,
			"adv_std": adv_std,
		}

	return {
		"policy_loss": accum_policy / total_batches,
		"value_loss": accum_value / total_batches,
		"entropy": accum_entropy / total_batches,
		"ratio_mean": ratio_mean_accum / total_batches,
		"ratio_std": ratio_std_accum / total_batches,
		"value_pred_mean": value_mean,
		"value_pred_std": value_std,
		"returns_mean": return_mean,
		"returns_std": return_std,
		"adv_mean": adv_mean,
		"adv_std": adv_std,
	}


class PPOAlgorithm(RLAlgorithm):
	requires_full_state = True

	def __init__(self, model: DVRPNet, optimizer: torch.optim.Optimizer, device: torch.device, args: Any) -> None:
		super().__init__(model, optimizer, device, args)
		# Shared-encoder pairwise graph critic over agent states (x,y,s,t) plus demand context
		self.critic = PairwiseGraphCritic(agent_dim=4, hidden_dim=max(128, model.d_model), global_dim=2 * model.d_model).to(device)
		self.critic.train()
		self.value_opt = torch.optim.AdamW(self.critic.parameters(), lr=args.value_lr)
		self.gamma = args.gamma
		self.gae_lambda = getattr(args, "gae_lambda", 0.95)
		self.use_gae = getattr(args, "use_gae", True)
		self.buffer = RolloutBuffer()

	def begin_episode(self, episode_idx: int) -> None:
		self.buffer.clear()

	def record_decision(self, record: DecisionRecord) -> None:
		if record.feats is None or record.agents is None or record.actions is None:
			raise ValueError("PPOAlgorithm requires full state tensors; set requires_full_state=True when capturing data.")
		if record.state_value is None:
			raise ValueError("PPOAlgorithm expects state_value to be filled when using GAE.")
		self.buffer.add(
			step_index=record.step_index,
			feats=record.feats,
			agents=record.agents,
			actions=record.actions,
			log_prob_sum=record.log_prob_sum,
			value=record.state_value,
			reward=float(record.reward),
			done=bool(record.done),
		)

	def end_episode(
		self,
		total_reward: float,
		rewards: List[float],
		dones: List[bool],
		env_stats: Dict[str, Any],
	) -> Dict[str, float]:
		if not rewards:
			# still run update based on whatever is in buffer (if any)
			if self.buffer.empty:
				return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
		stats = ppo_update(
			self.model,
			self.critic,
			self.optimizer,
			self.value_opt,
			self.buffer,
			self.args,
			self.device,
			self.args.lateness_lambda,
			self.gamma,
			gae_lambda=self.gae_lambda,
			use_gae=self.use_gae,
		)
		return stats
