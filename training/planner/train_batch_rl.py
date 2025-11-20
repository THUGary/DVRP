from __future__ import annotations

import argparse
import copy
import csv
import datetime
import os
import pathlib
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Ensure project root on sys.path (same logic as baseline script)
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
	_ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from configs import Config, get_default_config
from environment.env_tensor import TensorEnvObservation, TensorGridEnvironment
from agent.controller import RuleBasedController
from models.planner_model.model import DVRPNet, prepare_agents, prepare_features
from training.planner.rl_algorithms.sampling import select_targets_with_sampling
from training.planner import train_baseline_rl as baseline_rl

# Reuse helper utilities from the scalar baseline implementation
set_global_seed = baseline_rl.set_global_seed
ensure_log_file = baseline_rl.ensure_log_file
plot_training_curves = baseline_rl.plot_training_curves
prepare_val_seeds = baseline_rl.prepare_val_seeds
_checkpoint_dir_and_prefix = baseline_rl._checkpoint_dir_and_prefix


@dataclass
class BatchEpisodeResult:
	rewards: torch.Tensor
	log_prob_mean: Optional[torch.Tensor]
	entropy_mean: Optional[torch.Tensor]
	env_stats: List[Dict[str, float]]


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Vectorized REINFORCE training using TensorGridEnvironment")
	p.add_argument("--batches", type=int, default=200, help="Number of optimizer updates (outer loop)")
	p.add_argument(
		"--batch_size",
		type=int,
		default=4,
		help="Number of parallel environments rolled out per update (true batch size)",
	)
	p.add_argument("--max_demands", type=int, default=512, help="Max outstanding demands tracked per environment instance")
	p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the policy model")
	p.add_argument("--seed", type=int, default=0, help="Global random seed")
	p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
	p.add_argument("--ckpt_init", type=str, default=None, help="Optional checkpoint to warm start the policy model")
	p.add_argument(
		"--baseline_ckpt",
		type=str,
		default=None,
		help="(Unused) kept for parity with baseline script; baseline always clones the policy",
	)
	p.add_argument(
		"--save_best",
		type=str,
		default="checkpoints/planner_rl",
		help="Base filename (without suffix) for periodic checkpoints",
	)
	p.add_argument("--generator", type=str, choices=["rule", "net"], default="rule", help="Demand generator override")
	p.add_argument("--lateness_lambda", type=float, default=0.0, help="Soft lateness penalty during decode")
	p.add_argument("--target_queue_len", type=int, default=1, help="Length of autoregressive queue during rollout")
	p.add_argument("--history_window", type=int, default=16, help="Number of recent steps retained for history encoding")
	p.add_argument("--reward_log", type=str, default="runs/baseline_rl_rewards.csv", help="CSV log for batch rewards")
	p.add_argument("--update_cycle", type=int, default=20, help="Batches between validation cycles")
	p.add_argument("--val_num", type=int, default=15, help="Number of validation environments")
	p.add_argument(
		"--val_data_path",
		type=str,
		default="training/planner/data/baseline_val.pt",
		help="Path to cached validation seed list",
	)
	p.add_argument("--gen_val_data", action="store_true", help="Regenerate the validation seed dataset before training")
	p.add_argument("--entropy_coef", type=float, default=0.01, help="Initial entropy regularization coefficient λ")
	p.add_argument(
		"--entropy_decay",
		type=float,
		default=1.0,
		help="Multiplicative decay applied to λ after each batch (use <1.0 to decay)",
	)
	p.add_argument("--debug", action="store_true", help="Print per-step demand diagnostics for batch 0")
	p.add_argument("--deterministic", action="store_true", help="Enable deterministic CuDNN behavior")
	p.add_argument("--run_name", type=str, default=None, help="Run name for TensorBoard & plots")
	return p.parse_args()


def _make_generator_factory(cfg: Config):
	if cfg.generator_type == "net":
		from agent.generator.net_generator import NetDemandGenerator as GenClass
	else:
		from agent.generator import RuleBasedGenerator as GenClass

	def factory() -> object:
		params = copy.deepcopy(cfg.generator_params)
		return GenClass(cfg.width, cfg.height, **params)

	return factory


def build_tensor_env_from_cfg(
	cfg: Config,
	*,
	batch_size: int,
	device: torch.device,
	max_demands: int,
) -> TensorGridEnvironment:
	return TensorGridEnvironment(
		width=cfg.width,
		height=cfg.height,
		num_agents=cfg.num_agents,
		capacity=cfg.capacity,
		depot=cfg.depot,
		batch_size=batch_size,
		max_demands=max_demands,
		generator_factory=_make_generator_factory(cfg),
		device=device,
		include_service_time=cfg.include_service_time,
		max_time=cfg.max_time,
		max_end_time=cfg.max_end_time,
		expiry_penalty_scale=float(cfg.expiry_penalty_scale),
		switch_penalty_scale=float(cfg.switch_penalty_scale),
		capacity_reward_scale=float(cfg.capacity_reward_scale),
		wait_penalty_scale=float(cfg.wait_penalty_scale),
		move_penalty_scale=float(cfg.move_penalty_scale),
	)


def _tensor_obs_to_policy_inputs(
	obs: TensorEnvObservation,
	*,
	model: DVRPNet,
	device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
	time = obs.time.to(device)
	active = torch.logical_and(obs.demand_mask, obs.demands_start <= time.view(-1, 1))
	active_counts = active.sum(dim=1)
	max_active = int(active_counts.max().item()) if active_counts.numel() > 0 else 0
	batch_size = obs.time.size(0)
	if max_active <= 0:
		nodes = torch.zeros(batch_size, 0, 5, dtype=torch.float32, device=device)
		node_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
	else:
		nodes = torch.zeros(batch_size, max_active, 5, dtype=torch.float32, device=device)
		node_mask = torch.ones(batch_size, max_active, dtype=torch.bool, device=device)
		pos = obs.demands_pos.to(device=device, dtype=torch.float32)
		start = obs.demands_start.to(device=device, dtype=torch.float32)
		cap = obs.demands_capacity.to(device=device, dtype=torch.float32)
		end = obs.demands_end.to(device=device, dtype=torch.float32)
		for b in range(batch_size):
			idx = torch.nonzero(active[b], as_tuple=False).flatten()
			if idx.numel() == 0:
				continue
			use = idx[:max_active]
			used = int(use.numel())
			nodes[b, :used, 0] = pos[b, use, 0]
			nodes[b, :used, 1] = pos[b, use, 1]
			nodes[b, :used, 2] = start[b, use]
			nodes[b, :used, 3] = cap[b, use]
			nodes[b, :used, 4] = end[b, use]
			node_mask[b, :used] = False
	depot = obs.depot.to(torch.float32).unsqueeze(1).to(device)
	feats = prepare_features(
		nodes=nodes,
		node_mask=node_mask,
		depot=depot,
		d_model=model.d_model,
		device=device,
	)
	agents = torch.zeros(
		obs.agent_pos.size(0),
		obs.agent_pos.size(1),
		4,
		dtype=torch.float32,
		device=device,
	)
	agents[..., 0:2] = obs.agent_pos.to(torch.float32)
	agents[..., 2] = obs.agent_load.to(torch.float32)
	agents[..., 3] = time.float().unsqueeze(1)
	agents_tensor = prepare_agents(agents, device=device)
	return feats, agents_tensor, active


def _build_history_tensor(
	history: List[List[List[tuple[int, int]]]],
	device: torch.device,
	max_len_cap: int,
) -> torch.Tensor:
	batch = len(history)
	if batch == 0:
		return torch.zeros(0, device=device)
	num_agents = len(history[0]) if history[0] else 0
	max_len_cap = max(1, int(max_len_cap))
	max_len = max((len(seq) for env in history for seq in env), default=1)
	max_len = min(max_len, max_len_cap)
	tensor = torch.full((batch, num_agents, max_len, 2), -1.0, dtype=torch.float32, device=device)
	for b_idx, env_seq in enumerate(history):
		for a_idx, seq in enumerate(env_seq):
			start = max(0, len(seq) - max_len)
			trimmed = seq[start:]
			for t_idx, (x, y) in enumerate(trimmed):
				tensor[b_idx, a_idx, t_idx, 0] = float(x)
				tensor[b_idx, a_idx, t_idx, 1] = float(y)
	return tensor


def _extract_episode_stats(info: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, float]]:
	stats = [dict() for _ in range(batch_size)]
	if not info:
		return stats
	ep_stats = info.get("episode_stats")
	if not isinstance(ep_stats, dict):
		return stats
	for key, tensor in ep_stats.items():
		if not torch.is_tensor(tensor):
			continue
		values = tensor.detach().cpu().tolist()
		for idx in range(min(batch_size, len(values))):
			stats[idx][key] = float(values[idx])
	return stats


def _controller_actions(
	controller: RuleBasedController,
	current_pos: torch.Tensor,
	target_xy: torch.Tensor,
	alive_mask: torch.Tensor,
) -> torch.Tensor:
	batch, num_agents, _ = current_pos.shape
	actions = torch.zeros(batch, num_agents, 2, dtype=torch.long, device=current_pos.device)
	alive_bool = alive_mask > 0
	for b in range(batch):
		if not bool(alive_bool[b].item()):
			continue
		for a in range(num_agents):
			target_queue = deque()
			target_queue.append((int(target_xy[b, a, 0].item()), int(target_xy[b, a, 1].item())))
			x = int(current_pos[b, a, 0].item())
			y = int(current_pos[b, a, 1].item())
			dx, dy = controller.act((x, y), target_queue)
			actions[b, a, 0] = dx
			actions[b, a, 1] = dy
	return actions


def run_batch_episode(
	*,
	model: DVRPNet,
	env: TensorGridEnvironment,
	controller: RuleBasedController,
	args: argparse.Namespace,
	device: torch.device,
	seeds: Sequence[int],
	collect_traces: bool,
	action_selection: str,
	env_verbose: bool = False,
) -> BatchEpisodeResult:
	if seeds and len(seeds) != env.batch_size:
		raise ValueError(f"Expected {env.batch_size} seeds, got {len(seeds)}")
	batch_seed = seeds[0] if seeds else args.seed
	set_global_seed(batch_seed, device)
	ctx = torch.enable_grad() if collect_traces else torch.no_grad()
	history_limit = max(1, int(args.history_window))
	with ctx:
		observation = env.reset(seeds=seeds)
		history_positions: List[List[List[tuple[int, int]]]] = []
		history_targets: List[List[List[tuple[int, int]]]] = []
		for b in range(env.batch_size):
			agent_hist = []
			target_hist = []
			for a in range(env.num_agents):
				agent_hist.append([(int(observation.agent_pos[b, a, 0].item()), int(observation.agent_pos[b, a, 1].item()))])
				target_hist.append([(int(observation.depot[b, 0].item()), int(observation.depot[b, 1].item()))])
			history_positions.append(agent_hist)
			history_targets.append(target_hist)

		done_mask = torch.zeros(env.batch_size, dtype=torch.bool, device=device)
		total_reward = torch.zeros(env.batch_size, dtype=torch.float32, device=device)
		log_prob_terms: List[torch.Tensor] = []
		entropy_terms: List[torch.Tensor] = []
		alive_history: List[torch.Tensor] = []
		last_info: Dict[str, torch.Tensor] = {}

		while True:
			if torch.all(done_mask):
				break
			alive_mask = (~done_mask).float()
			alive_history.append(alive_mask)

			feats, agents_tensor, _ = _tensor_obs_to_policy_inputs(observation, model=model, device=device)
			hp = _build_history_tensor(history_positions, device, args.history_window)
			ht = _build_history_tensor(history_targets, device, args.history_window)

			sel, dest_xy, log_probs, entropies, _value, _queue_idx, _queue_coords = select_targets_with_sampling(
				model=model,
				feats=feats,
				agents_tensor=agents_tensor,
				lateness_lambda=args.lateness_lambda,
				critic=None,
				history_positions=hp,
				history_target_coords=ht,
				target_queue_len=args.target_queue_len,
				selection_strategy=action_selection,
			)

			if collect_traces:
				log_prob_terms.append(log_probs.sum(dim=1) * alive_mask)
				entropy_terms.append(entropies.sum(dim=1) * alive_mask)

			actions = _controller_actions(controller, observation.agent_pos, dest_xy, alive_mask)
			observation, reward, done_step, info = env.step(actions)
			reward = reward.to(torch.float32)
			total_reward += reward * alive_mask
			done_mask = torch.logical_or(done_mask, done_step)
			last_info = info

			alive_bool = alive_mask > 0
			for b in range(env.batch_size):
				if not bool(alive_bool[b].item()):
					continue
				for a in range(env.num_agents):
					history_targets[b][a].append((int(dest_xy[b, a, 0].item()), int(dest_xy[b, a, 1].item())))
					if len(history_targets[b][a]) > history_limit:
						history_targets[b][a].pop(0)
					history_positions[b][a].append(
						(int(observation.agent_pos[b, a, 0].item()), int(observation.agent_pos[b, a, 1].item()))
					)
					if len(history_positions[b][a]) > history_limit:
						history_positions[b][a].pop(0)

		log_prob_mean = None
		entropy_mean = None
		if collect_traces and log_prob_terms:
			participation = torch.stack(alive_history).sum(dim=0).clamp(min=1.0)
			log_prob_sum = torch.stack(log_prob_terms).sum(dim=0)
			log_prob_mean = log_prob_sum / participation
			if entropy_terms:
				entropy_sum = torch.stack(entropy_terms).sum(dim=0)
				entropy_mean = entropy_sum / participation
			else:
				entropy_mean = torch.zeros_like(log_prob_mean)

		stats = _extract_episode_stats(last_info, env.batch_size)
		return BatchEpisodeResult(rewards=total_reward, log_prob_mean=log_prob_mean, entropy_mean=entropy_mean, env_stats=stats)


def evaluate_model_on_dataset(
	*,
	model: DVRPNet,
	cfg: Config,
	args: argparse.Namespace,
	device: torch.device,
	seeds: List[int],
) -> float:
	"""Evaluate `model` on the provided seeds using batched tensor environments."""
	if not seeds:
		return 0.0
	prev_training = model.training
	model.eval()
	controller = RuleBasedController(**cfg.controller_params)
	batch_size = max(1, int(getattr(args, "batch_size", 1)))
	max_demands = int(getattr(args, "max_demands", cfg.max_time * 10))
	total_reward = 0.0
	total_episodes = 0
	idx = 0
	while idx < len(seeds):
		chunk = seeds[idx : idx + batch_size]
		chunk_size = len(chunk)
		env = build_tensor_env_from_cfg(
			cfg,
			batch_size=chunk_size,
			device=device,
			max_demands=max_demands,
		)
		result = run_batch_episode(
			model=model,
			env=env,
			controller=controller,
			args=args,
			device=device,
			seeds=chunk,
			collect_traces=False,
			action_selection="greedy",
			env_verbose=False,
		)
		total_reward += float(result.rewards.sum().item())
		total_episodes += chunk_size
		idx += chunk_size
	if prev_training:
		model.train()
	return total_reward / max(1, total_episodes)


def main() -> None:
	args = parse_args()
	device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
	if args.deterministic and device.type == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		try:
			torch.use_deterministic_algorithms(True)
		except Exception:
			pass

	cfg = get_default_config()
	if args.generator:
		cfg.generator_type = args.generator
	val_seeds, val_path = prepare_val_seeds(args)
	print(f"[VAL] Using {len(val_seeds)} validation seeds from {val_path}")

	env_policy = build_tensor_env_from_cfg(cfg, batch_size=args.batch_size, device=device, max_demands=args.max_demands)
	env_baseline = build_tensor_env_from_cfg(cfg, batch_size=args.batch_size, device=device, max_demands=args.max_demands)
	controller_policy = RuleBasedController(**cfg.controller_params)
	controller_baseline = RuleBasedController(**cfg.controller_params)

	model = DVRPNet(
		d_model=cfg.model_planner_params.get("d_model", 128),
		nhead=cfg.model_planner_params.get("nhead", 8),
		nlayers=cfg.model_planner_params.get("nlayers", 2),
	).to(device)
	model.train()

	if args.ckpt_init and os.path.exists(args.ckpt_init):
		blob = torch.load(args.ckpt_init, map_location=device)
		state = blob.get("model", blob)
		missing, unexpected = model.load_state_dict(state, strict=False)
		print(f"[INIT] Warm start from {args.ckpt_init} (missing={len(missing)}, unexpected={len(unexpected)})")
	else:
		print("[INIT] Training from random initialization")

	baseline_model = DVRPNet(
		d_model=cfg.model_planner_params.get("d_model", 128),
		nhead=cfg.model_planner_params.get("nhead", 8),
		nlayers=cfg.model_planner_params.get("nlayers", 2),
	).to(device)
	baseline_model.load_state_dict(model.state_dict())
	baseline_model.eval()
	print("[BASELINE] Initialized from policy parameters")

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	run_name = args.run_name if args.run_name else f"planner_batch_rl_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
	log_dir = os.path.join(_ROOT, "training/planner/runs", run_name)
	writer = SummaryWriter(log_dir=log_dir)
	print(f"[TB] TensorBoard logging to {log_dir}")

	if args.reward_log == "runs/baseline_rl_rewards.csv":
		args.reward_log = os.path.join(log_dir, "rewards.csv")
	ensure_log_file(args.reward_log, ["batch", "policy_reward", "baseline_reward", "advantage"])
	checkpoint_dir, checkpoint_prefix = _checkpoint_dir_and_prefix(args)

	def dump_checkpoint(batch_idx: int) -> Optional[pathlib.Path]:
		if batch_idx <= 0:
			return None
		path = checkpoint_dir / f"{checkpoint_prefix}_{batch_idx}.pt"
		torch.save({"model": model.state_dict(), "batch": batch_idx}, path)
		print(f"[SAVE] checkpoint => {path}")
		return path

	last_completed_batch = 0
	latest_checkpoint: Optional[pathlib.Path] = None

	try:
		for batch_idx in range(1, args.batches + 1):
			model.train()
			optimizer.zero_grad()

			seed_base = args.seed + (batch_idx - 1) * args.batch_size
			seed_vector = [seed_base + i for i in range(args.batch_size)]

			policy_result = run_batch_episode(
				model=model,
				env=env_policy,
				controller=controller_policy,
				args=args,
				device=device,
				seeds=seed_vector,
				collect_traces=True,
				action_selection="stochastic",
				env_verbose=False,
			)

			baseline_model.eval()
			baseline_result = run_batch_episode(
				model=baseline_model,
				env=env_baseline,
				controller=controller_baseline,
				args=args,
				device=device,
				seeds=seed_vector,
				collect_traces=False,
				action_selection="greedy",
				env_verbose=False,
			)

			policy_rewards = policy_result.rewards.detach()
			baseline_rewards = baseline_result.rewards.detach()
			adv_tensor = policy_rewards - baseline_rewards
			log_prob_mean = policy_result.log_prob_mean
			if log_prob_mean is None:
				continue
			entropy_mean = policy_result.entropy_mean if policy_result.entropy_mean is not None else torch.zeros_like(log_prob_mean)
			current_entropy_coef = max(0.0, args.entropy_coef * (args.entropy_decay ** max(0, batch_idx - 1)))

			loss_rl = -(adv_tensor.to(device) * log_prob_mean).mean()
			loss_entropy = -current_entropy_coef * entropy_mean.mean()
			loss = loss_rl + loss_entropy
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			avg_policy_reward = float(policy_rewards.mean().item())
			avg_baseline_reward = float(baseline_rewards.mean().item())
			avg_advantage = float(adv_tensor.mean().item())
			avg_loss = float(loss.item())
			avg_entropy = float(entropy_mean.mean().item()) if entropy_mean is not None else 0.0

			writer.add_scalar("Train/Loss_Total", avg_loss, batch_idx)
			writer.add_scalar("Train/Entropy_Coef", current_entropy_coef, batch_idx)
			writer.add_scalar("Train/Entropy_Raw", avg_entropy, batch_idx)
			writer.add_scalar("Reward/Policy", avg_policy_reward, batch_idx)
			writer.add_scalar("Reward/Baseline", avg_baseline_reward, batch_idx)
			writer.add_scalar("Reward/Advantage", avg_advantage, batch_idx)

			if policy_result.env_stats:
				aggregated: Dict[str, List[float]] = {}
				for entry in policy_result.env_stats:
					for key, value in entry.items():
						aggregated.setdefault(key, []).append(float(value))
				for key, values in aggregated.items():
					writer.add_scalar(f"EnvStats/{key}", float(np.mean(values)), batch_idx)

			if args.reward_log:
				with open(args.reward_log, "a", newline="") as fh:
					csv.writer(fh).writerow([batch_idx, avg_policy_reward, avg_baseline_reward, avg_advantage])

			status = "encourage" if avg_advantage >= 0 else "penalize"
			print(
				f"[BATCH {batch_idx:04d}] policy={avg_policy_reward:.2f} baseline={avg_baseline_reward:.2f} "
				f"adv={avg_advantage:.2f} action={status} loss={avg_loss:.4f} "
				f"entropy_term={current_entropy_coef * avg_entropy:.4f}"
			)

			if batch_idx % 100 == 0:
				latest_checkpoint = dump_checkpoint(batch_idx) or latest_checkpoint

			if batch_idx % args.update_cycle == 0:
				avg_policy = evaluate_model_on_dataset(
					model=model,
					cfg=cfg,
					args=args,
					device=device,
					seeds=val_seeds,
				)
				avg_baseline = evaluate_model_on_dataset(
					model=baseline_model,
					cfg=cfg,
					args=args,
					device=device,
					seeds=val_seeds,
				)
				print(f"[VAL][BATCH {batch_idx:04d}] policy_avg={avg_policy:.2f} baseline_avg={avg_baseline:.2f}")
				writer.add_scalar("Val/Policy_Avg_Reward", avg_policy, batch_idx)
				writer.add_scalar("Val/Baseline_Avg_Reward", avg_baseline, batch_idx)
				if avg_policy > avg_baseline:
					baseline_model.load_state_dict(model.state_dict())
					baseline_model.eval()
					print(f"[BASELINE] Updated baseline model at BATCH {batch_idx:04d}")
					writer.add_scalar("Baseline/Updates", 1, batch_idx)
				else:
					writer.add_scalar("Baseline/Updates", 0, batch_idx)

			last_completed_batch = batch_idx

	except KeyboardInterrupt:
		print("[INTERRUPT] Training interrupted; final checkpoint will be saved.")
	finally:
		final_path = dump_checkpoint(last_completed_batch)
		if final_path is not None:
			latest_checkpoint = final_path
		if args.reward_log and os.path.exists(args.reward_log):
			plot_training_curves(args.reward_log, log_dir, run_name)
		writer.close()

	if latest_checkpoint is not None:
		print(f"[DONE] Latest checkpoint saved to {latest_checkpoint}")
	else:
		print("[DONE] Training finished but no batches were completed; no checkpoint saved.")


if __name__ == "__main__":
	main()
