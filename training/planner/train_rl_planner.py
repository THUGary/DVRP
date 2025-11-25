from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple
from datetime import datetime
import csv

# Ensure project root on sys.path from nested training directory
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter

plt.switch_backend("Agg")

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.controller import RuleBasedController

from models.planner_model.model import DVRPNet, prepare_features, prepare_agents
from training.planner.rl_algorithms import (
    ReinforceAlgorithm,
    PPOAlgorithm,
    RLAlgorithm,
    DecisionRecord,
    POMOAlgorithm,
)
from training.planner.rl_algorithms.sampling import select_targets_with_sampling, detach_feats


ALGORITHM_REGISTRY = {
    "reinforce": ReinforceAlgorithm,
    "ppo": PPOAlgorithm,
    "pomo": POMOAlgorithm,
}


def build_algorithm(name: str, model: DVRPNet, optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> RLAlgorithm:
    algo_cls = ALGORITHM_REGISTRY.get(name)
    if algo_cls is None:
        raise ValueError(f"Unsupported RL algorithm '{name}'. Available: {sorted(ALGORITHM_REGISTRY.keys())}")
    return algo_cls(model=model, optimizer=optimizer, device=device, args=args)


def format_metrics(stats: Dict[str, float]) -> str:
    if not stats:
        return ""
    parts = []
    for key, value in stats.items():
        try:
            parts.append(f"{key}={value:.3f}")
        except (TypeError, ValueError):
            parts.append(f"{key}={value}")
    return " " + " ".join(parts)


def build_env_from_cfg(cfg: Config) -> GridEnvironment:
    """
    Build and return a GridEnvironment using values from a Config object.

    This helper picks the demand generator class (net vs rule) according to
    `cfg.generator_type` and then constructs the environment with the standard
    parameters from the config. It also sets `env.num_agents` to match the
    config for downstream code that reads this attribute.

    Args:
        cfg: configuration object returned by `get_default_config()`.

    Returns:
        An instance of `GridEnvironment` configured according to `cfg`.
    """
    # choose generator class by config
    if cfg.generator_type == "net":
        from agent.generator.net_generator import NetDemandGenerator as GenClass
    else:
        from agent.generator import RuleBasedGenerator as GenClass

    gen = GenClass(cfg.width, cfg.height, **cfg.generator_params)
    env = GridEnvironment(
        width=cfg.width,
        height=cfg.height,
        num_agents=cfg.num_agents,
        capacity=cfg.capacity,
        depot=cfg.depot,
        generator=gen,
        max_time=cfg.max_time,
        expiry_penalty_scale=float(getattr(cfg, "expiry_penalty_scale", 5.0)),
        switch_penalty_scale=float(getattr(cfg, "switch_penalty_scale", 0.01)),
        capacity_reward_scale=float(getattr(cfg, "capacity_reward_scale", 10.0)),
        exploration_history_n=int(getattr(cfg, "exploration_history_n", 0)),
        exploration_penalty_scale=float(getattr(cfg, "exploration_penalty_scale", 0.0)),
        wait_penalty_scale=float(getattr(cfg, "wait_penalty_scale", 0.001)),
        distance_penalty_base=float(getattr(cfg, "distance_penalty_base", 0.0)),
        distance_penalty_min_dist=float(getattr(cfg, "distance_penalty_min_dist", 1.0)),
        move_penalty_scale=float(getattr(cfg, "move_penalty_scale", 0.0)),
        approach_bonus_scale=float(getattr(cfg, "approach_bonus_scale", 0.0)),
        approach_bonus_max_dist=float(getattr(cfg, "approach_bonus_max_dist", 0.0)),
        max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
        include_service_time=bool(getattr(cfg, "include_service_time", False)),
    )
    env.num_agents = cfg.num_agents
    return env 


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the RL fine-tuning script.

    Returns:
        An argparse.Namespace containing script options and hyperparameters.
    """
    p = argparse.ArgumentParser(description="RL fine-tuning for DVRPNet (policy gradient)")
    p.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    # If the flag is omitted entirely we do NOT warm-start.
    # If the flag is present but no path is given, use the default checkpoint path.
    p.add_argument("--ckpt_init", nargs='?', const="checkpoints/planner_20_2_10.pt", default=None,
                   help="Optional: initial planner checkpoint to warm start (flag present with no value uses default path)")
    p.add_argument("--save_best", type=str, default="checkpoints/planner_rl_best.pt", help="Path to save the best-performing RL checkpoint")
    p.add_argument("--generator", type=str, choices=["rule", "net"], default="rule", help="Override generator type for RL training")
    p.add_argument("--lateness_lambda", type=float, default=0.0, help="Soft lateness penalty used during decode")
    p.add_argument("--reward_log", type=str, default="runs/rl_rewards.csv", help="CSV file to log per-episode rewards")
    p.add_argument("--reward_plot", type=str, default="runs/rl_rewards.png", help="Path to save reward curve plot")
    p.add_argument("--ppo_diag_plot", type=str, default="runs/ppo_diagnostics.png", help="Path to save PPO diagnostic plot (ratio mean/std + value loss)")
    p.add_argument("--algo", type=str, default="reinforce", choices=["reinforce", "ppo", "pomo"], help="Policy gradient algorithm to use")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns")
    p.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per episode")
    p.add_argument("--ppo_batch_size", type=int, default=32, help="Mini-batch size for PPO updates")
    p.add_argument("--ppo_clip", type=float, default=0.2, help="Clipping epsilon for PPO objective")
    p.add_argument("--value_lr", type=float, default=1e-4, help="Learning rate for the value critic (PPO)")
    p.add_argument("--value_coef", type=float, default=0.5, help="Weight for value loss in PPO")
    p.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient for PPO")
    p.add_argument("--normalize_adv", action="store_true", help="Normalize advantages before PPO update")
    p.add_argument("--target_queue_len", type=int, default=1, help="Length of autoregressive target queue to propose per agent")
    p.add_argument("--debug", action="store_true", help="Enable per-step debug printing of observed demands and new demands")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic torch/CuDNN behavior (may reduce performance)")
    p.add_argument("--tb_logdir", type=str, default="runs/tb", help="TensorBoard log directory (empty string to disable)")
    p.add_argument("--num_agents", type=int, default=None, help="Override number of agents used during RL training (defaults to config value)")
    p.add_argument("--pomo_size", type=int, default=1, help="Number of POMO replicas (self-competition rollouts) per episode")
    p.add_argument("--pomo_entropy_coef", type=float, default=0.0, help="Entropy bonus coefficient for POMO updates")
    p.add_argument("--pomo_grad_clip", type=float, default=1.0, help="Gradient max-norm for POMO updates")
    p.add_argument("--pomo_seed_stride", type=int, default=9973, help="Stride applied to torch seed between POMO replicas")
    p.add_argument("--pomo_baseline", type=str, choices=["mean", "leave_one_out"], default="leave_one_out",
                   help="Baseline strategy for POMO advantage computation")
    p.add_argument("--critic_diag_dir", type=str, default="runs/planner",
                   help="Base directory to store critic diagnostic CSV/plots (disabled when empty)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Seed global RNGs for reproducibility. We also support a stricter
    # deterministic mode that configures cuDNN; note this may reduce
    # performance or cause errors for some ops.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    if args.deterministic and device.type == "cuda":
        # Make cuDNN deterministic (may slow down and restrict some ops).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not have this; ignore if unavailable.
            pass

    # Config & env
    cfg = get_default_config()
    if args.generator:
        cfg.generator_type = args.generator
    if args.num_agents is not None:
        cfg.num_agents = max(1, int(args.num_agents))
    # IMPORTANT: depot will be randomized in run-like script; here we keep default depot

    env = build_env_from_cfg(cfg)
    controller = RuleBasedController(**cfg.controller_params)

    # Build model and warm start
    model = DVRPNet(
        d_model=cfg.model_planner_params.get("d_model", 128),
        nhead=cfg.model_planner_params.get("nhead", 8),
        nlayers=cfg.model_planner_params.get("nlayers", 2),
    ).to(device)
    model.train()

    # Only warm-start if the flag was provided on the command line. If the
    # flag was omitted (args.ckpt_init is None) we explicitly do NOT warm-start.
    if args.ckpt_init is None:
        print("[RL] No warm-start requested; training from random init.")
    else:
        # Flag present: attempt to load the specified checkpoint (or default
        # path if the flag was given without a value).
        if os.path.exists(args.ckpt_init):
            blob = torch.load(args.ckpt_init, map_location=device)
            state = blob.get("model", blob)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[RL] Warm start from {args.ckpt_init} (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"[RL] Warm-start requested but checkpoint not found at {args.ckpt_init}; training from random init.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    reward_history: List[float] = []
    best_return = float("-inf")

    if args.reward_log:
        reward_log_dir = os.path.dirname(args.reward_log) or "."
        os.makedirs(reward_log_dir, exist_ok=True)
        if not os.path.exists(args.reward_log):
            with open(args.reward_log, "w", newline="") as fh:
                csv_writer = csv.writer(fh)
                csv_writer.writerow(["episode", "return"])

    if args.reward_plot:
        reward_plot_dir = os.path.dirname(args.reward_plot) or "."
        os.makedirs(reward_plot_dir, exist_ok=True)

    if args.save_best:
        save_best_dir = os.path.dirname(args.save_best) or "."
        os.makedirs(save_best_dir, exist_ok=True)

    if getattr(args, "ppo_diag_plot", None):
        ppo_diag_dir = os.path.dirname(args.ppo_diag_plot) or "."
        os.makedirs(ppo_diag_dir, exist_ok=True)

    writer: SummaryWriter | None = None
    if getattr(args, "tb_logdir", None):
        if len(args.tb_logdir.strip()) > 0:
            os.makedirs(args.tb_logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=args.tb_logdir)
    rl_algo = build_algorithm(args.algo, model, opt, device, args)

    ratio_mean_history: List[float] = []
    ratio_std_history: List[float] = []
    value_loss_history: List[float] = []
    critic_log_rows: List[Dict[str, float]] = []
    pomo_size = max(1, int(getattr(args, "pomo_size", 1)))
    pomo_seed_stride = max(1, int(getattr(args, "pomo_seed_stride", 9973)))
    if args.algo != "pomo":
        pomo_size = 1

    for ep in range(1, args.episodes + 1):
        rl_algo.begin_group(ep)
        seed_ep = int(args.seed + ep)
        group_rewards: List[float] = []
        best_rep_reward = float("-inf")
        best_env_stats: Dict[str, Any] = {}
        best_depot_counts = (0, 0)
        last_episode_stats: Dict[str, float] = {}

        for pomo_rank in range(pomo_size):
            rl_algo.begin_episode((ep - 1) * pomo_size + pomo_rank + 1)

            # Ensure environment randomness is identical across replicas
            random.seed(seed_ep)
            np.random.seed(seed_ep)
            torch.manual_seed(seed_ep)
            if device.type == "cuda":
                try:
                    torch.cuda.manual_seed_all(seed_ep)
                except Exception:
                    pass

            obs = env.reset(seed=seed_ep)

            # Policy randomness diverges per POMO replica through torch seeds only
            torch_seed = seed_ep + pomo_rank * pomo_seed_stride
            torch.manual_seed(torch_seed)
            if device.type == "cuda":
                try:
                    torch.cuda.manual_seed_all(torch_seed)
                except Exception:
                    pass

            total_reward = 0.0
            rewards_all: List[float] = []
            dones_all: List[bool] = []
            prev_demands: List[Tuple[int, int, int, int, int]] = []
            done = False
            hist_pos: List[List[Tuple[int, int]]] = []
            hist_targets: List[List[Tuple[int, int]]] = []
            depot_xy0 = (int(obs["depot"][0]), int(obs["depot"][1]))
            for (x, y, s) in obs["agent_states"]:
                hist_pos.append([(int(x), int(y))])
                hist_targets.append([depot_xy0])
            depot_select_count = 0
            total_select_count = 0
            while not done:
                nodes_list = obs["demands"]
                step_idx = len(rewards_all)
                try:
                    new_demands = [d for d in nodes_list if d not in prev_demands]
                except Exception:
                    new_demands = list(nodes_list)
                if args.debug:
                    prefix = f"[EP {ep:04d} R{pomo_rank:02d} STEP {step_idx}]" if pomo_size > 1 else f"[EP {ep:04d} STEP {step_idx}]"
                    if nodes_list:
                        print(f"{prefix} demands={len(nodes_list)} new={len(new_demands)}")
                    else:
                        print(f"{prefix} no demands seen")
                N = len(nodes_list)
                node_mask = [False] * N
                depot = [obs["depot"]]

                if N == 0:
                    actions = [(0, 0) for _ in obs["agent_states"]]
                    next_obs, reward, done, _ = env.step(actions)
                    reward_val = float(reward)
                    total_reward += reward_val
                    rewards_all.append(reward_val)
                    dones_all.append(done)
                    prev_demands = list(nodes_list)
                    obs = next_obs
                    continue

                feats = prepare_features(nodes=[nodes_list], node_mask=[node_mask], depot=[depot], d_model=model.d_model, device=device)
                agents = [(x, y, s, obs["time"]) for (x, y, s) in obs["agent_states"]]
                agents_t = prepare_agents([agents], device=device)

                T_pos = max(len(h) for h in hist_pos)
                T_tgt = max(len(h) for h in hist_targets)
                T = max(T_pos, T_tgt)
                A = len(hist_pos)
                hp = torch.full((1, A, T, 2), -1, dtype=torch.float32, device=device)
                ht = torch.full((1, A, T, 2), -1, dtype=torch.float32, device=device)
                for a_idx, (seq_pos, seq_tgt) in enumerate(zip(hist_pos, hist_targets)):
                    for t_idx, (px, py) in enumerate(seq_pos):
                        hp[0, a_idx, t_idx, 0] = float(px)
                        hp[0, a_idx, t_idx, 1] = float(py)
                    for t_idx, (tx, ty) in enumerate(seq_tgt):
                        ht[0, a_idx, t_idx, 0] = float(tx)
                        ht[0, a_idx, t_idx, 1] = float(ty)

                critic_module = getattr(rl_algo, "critic", None)
                sel, dest_xy, log_probs, entropies, state_value, queue_indices, queue_coords = select_targets_with_sampling(
                    model=model,
                    feats=feats,
                    agents_tensor=agents_t,
                    lateness_lambda=args.lateness_lambda,
                    critic=critic_module,
                    history_positions=hp,
                    history_target_coords=ht,
                    target_queue_len=args.target_queue_len,
                )

                if args.debug:
                    B, A = sel.shape
                    depot_sel = int((sel == 0).sum().item())
                    total = B * A
                    prefix = f"[EP {ep:04d} R{pomo_rank:02d} STEP {step_idx}]" if pomo_size > 1 else f"[EP {ep:04d} STEP {step_idx}]"
                    print(f"{prefix} depot_ratio={depot_sel}/{total} = {depot_sel/float(max(1,total)):.2f}")

                actions: List[Tuple[int, int]] = []
                for i, (x, y, s) in enumerate(obs["agent_states"]):
                    tx, ty = int(dest_xy[0, i, 0].item()), int(dest_xy[0, i, 1].item())
                    q = deque()
                    q.append((tx, ty))
                    actions.append(controller.act((x, y), q))

                next_obs, reward, done, _ = env.step(actions)
                reward_val = float(reward)
                total_reward += reward_val

                log_prob_sum = log_probs.sum()
                rewards_all.append(reward_val)
                dones_all.append(done)
                depot_select_count += int((sel == 0).sum().item())
                total_select_count += sel.numel()

                record = DecisionRecord(
                    step_index=len(rewards_all) - 1,
                    log_prob_sum=log_prob_sum,
                    reward=reward_val,
                    done=done,
                    entropy_sum=entropies.sum(),
                )
                if rl_algo.requires_full_state:
                    record.feats = detach_feats(feats)
                    record.agents = agents_t.detach().cpu().clone()
                    record.actions = sel.detach().cpu().clone()
                    record.state_value = state_value.detach().cpu().clone() if state_value is not None else None
                    record.history_positions = hp.detach().cpu().clone()
                    record.history_targets = ht.detach().cpu().clone()
                    record.queue_indices = queue_indices.detach().cpu().clone()
                    record.queue_coords = queue_coords.detach().cpu().clone()
                rl_algo.record_decision(record)

                obs = next_obs
                prev_demands = list(nodes_list)
                hist_pos = [seq + [(int(x), int(y))] for seq, (x, y, s) in zip(hist_pos, obs["agent_states"]) ]
                hist_targets = [
                    seq + [(int(dest_xy[0, a, 0].item()), int(dest_xy[0, a, 1].item()))]
                    for a, seq in enumerate(hist_targets)
                ]

            env_stats = getattr(env, "_episode_stats", {})
            stats = rl_algo.end_episode(total_reward, rewards_all, dones_all, env_stats)
            last_episode_stats = stats or {}
            group_rewards.append(total_reward)
            if total_reward > best_rep_reward:
                best_rep_reward = total_reward
                best_env_stats = dict(env_stats)
                best_depot_counts = (depot_select_count, total_select_count)

        group_stats = rl_algo.end_group() or {}
        combined_stats: Dict[str, float] = {}
        if last_episode_stats:
            combined_stats.update(last_episode_stats)
        if group_stats:
            combined_stats.update(group_stats)

        avg_reward = float(sum(group_rewards) / len(group_rewards)) if group_rewards else 0.0
        reward_history.append(avg_reward)
        if args.reward_log:
            with open(args.reward_log, "a", newline="") as fh:
                csv_writer = csv.writer(fh)
                csv_writer.writerow([ep, avg_reward])

        msg = format_metrics(combined_stats)
        if pomo_size > 1:
            best_display = best_rep_reward if best_rep_reward != float("-inf") else avg_reward
            print(f"[EP {ep:04d}] avg={avg_reward:.5f} best={best_display:.5f} reps={pomo_size}{msg}")
        else:
            print(f"[EP {ep:04d}] return={avg_reward:.5f}{msg}")

        if args.algo == "ppo" and combined_stats:
            if "ratio_mean" in combined_stats:
                ratio_mean_history.append(float(combined_stats["ratio_mean"]))
            if "ratio_std" in combined_stats:
                ratio_std_history.append(float(combined_stats["ratio_std"]))
            if "value_loss" in combined_stats:
                value_loss_history.append(float(combined_stats["value_loss"]))
        if args.algo == "ppo":
            critic_row = {
                "episode": float(ep),
                "avg_reward": float(avg_reward),
                "best_reward": float(episode_best),
                "value_loss": float(combined_stats.get("value_loss", 0.0) or 0.0),
                "value_pred_mean": float(combined_stats.get("value_pred_mean", 0.0) or 0.0),
                "value_pred_std": float(combined_stats.get("value_pred_std", 0.0) or 0.0),
                "returns_mean": float(combined_stats.get("returns_mean", 0.0) or 0.0),
                "returns_std": float(combined_stats.get("returns_std", 0.0) or 0.0),
                "adv_mean": float(combined_stats.get("adv_mean", 0.0) or 0.0),
                "adv_std": float(combined_stats.get("adv_std", 0.0) or 0.0),
                "ratio_mean": float(combined_stats.get("ratio_mean", 0.0) or 0.0),
                "ratio_std": float(combined_stats.get("ratio_std", 0.0) or 0.0),
            }
            critic_log_rows.append(critic_row)

        episode_best = best_rep_reward if best_rep_reward != float("-inf") else avg_reward
        if episode_best > best_return:
            best_return = episode_best
            if args.save_best:
                torch.save({"model": model.state_dict(), "episode": ep, "return": episode_best}, args.save_best)
                print(f"[RL] new best checkpoint saved => {args.save_best} (return={episode_best:.5f})")

        if writer is not None:
            env_stats_src = best_env_stats if best_env_stats else getattr(env, "_episode_stats", {})
            demand_count = env_stats_src.get("demand_count", 0)
            demand_capacity = env_stats_src.get("demand_capacity", 0.0)
            served_count = env_stats_src.get("served_count", 0)
            served_capacity = env_stats_src.get("served_capacity", 0.0)
            expired_capacity = env_stats_src.get("expired_capacity", 0.0)
            capacity_reward_term = env_stats_src.get("capacity_reward_term", 0.0)
            expired_penalty_mag = env_stats_src.get("expired_penalty", 0.0)
            switch_penalty_term = env_stats_src.get("switch_penalty", 0.0)
            exploration_penalty_value = env_stats_src.get("exploration_penalty_value", 0.0)
            pairwise_penalty_value = env_stats_src.get("pairwise_penalty_value", 0.0)
            served_ratio = (served_capacity / demand_capacity) if demand_capacity > 1e-9 else 0.0
            best_depot, total_depot = best_depot_counts
            depot_ratio = (best_depot / total_depot) if total_depot > 0 else 0.0
            expiry_penalty_total = -expired_penalty_mag
            writer.add_scalar("episode/return", avg_reward, ep)
            writer.add_scalar("demand/count", demand_count, ep)
            writer.add_scalar("demand/capacity_total", demand_capacity, ep)
            writer.add_scalar("served/count", served_count, ep)
            writer.add_scalar("served/capacity_served", served_capacity, ep)
            writer.add_scalar("expired/capacity", expired_capacity, ep)
            writer.add_scalar("ratio/served_capacity_ratio", served_ratio, ep)
            writer.add_scalar("ratio/depot_ratio", depot_ratio, ep)
            if capacity_reward_term is not None:
                writer.add_scalar("reward_parts/capacity_reward_term", capacity_reward_term, ep)
            if expired_penalty_mag is not None:
                writer.add_scalar("reward_parts/expiry_penalty", expiry_penalty_total, ep)
            if switch_penalty_term is not None:
                writer.add_scalar("reward_parts/switch_penalty_term", switch_penalty_term, ep)
            if exploration_penalty_value is not None:
                writer.add_scalar("reward_parts/exploration_penalty_value", exploration_penalty_value, ep)
            if pairwise_penalty_value is not None:
                writer.add_scalar("reward_parts/pairwise_penalty_value", pairwise_penalty_value, ep)
            writer.flush()

    if reward_history and args.reward_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(reward_history) + 1), reward_history, label="Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("RL Training Reward")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(args.reward_plot)
        plt.close()
        print(f"[RL] reward curve saved => {args.reward_plot}")

    critic_diag_run_dir: str | None = None
    if critic_log_rows and getattr(args, "critic_diag_dir", "").strip():
        base_dir = args.critic_diag_dir.strip()
        os.makedirs(base_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        critic_diag_run_dir = os.path.join(base_dir, f"critic_{stamp}")
        os.makedirs(critic_diag_run_dir, exist_ok=True)
        csv_path = os.path.join(critic_diag_run_dir, "critic_metrics.csv")
        fieldnames = list(critic_log_rows[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(critic_log_rows)
        print(f"[RL] critic metrics saved => {csv_path}")

        def series(key: str) -> Tuple[List[float], List[float]]:
            xs: List[float] = []
            ys: List[float] = []
            for row in critic_log_rows:
                val = row.get(key)
                if val is None:
                    continue
                xs.append(row.get("episode", 0.0))
                ys.append(val)
            return xs, ys

        plt.figure(figsize=(8, 4))
        plotted = False
        for key, label in [
            ("value_loss", "Value Loss"),
            ("returns_mean", "Returns Mean"),
            ("value_pred_mean", "Value Prediction Mean"),
        ]:
            xs, ys = series(key)
            if xs:
                plt.plot(xs, ys, label=label)
                plotted = True
        if plotted:
            plt.xlabel("Episode")
            plt.ylabel("Metric Value")
            plt.title("Critic Diagnostics")
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.legend(loc="best")
            plt.tight_layout()
            plot_path = os.path.join(critic_diag_run_dir, "critic_metrics.png")
            plt.savefig(plot_path)
            print(f"[RL] critic diagnostics saved => {plot_path}")
        plt.close()

    if args.algo == "ppo" and ratio_mean_history and args.ppo_diag_plot:
        episodes_logged = range(1, len(ratio_mean_history) + 1)
        plt.figure(figsize=(8, 4))
        ax1 = plt.gca()
        ax1.plot(episodes_logged, ratio_mean_history, label="ratio_mean", color="tab:blue")
        if ratio_std_history:
            ax1.plot(episodes_logged[:len(ratio_std_history)], ratio_std_history, label="ratio_std", color="tab:orange")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Ratio stats")
        ax1.grid(True, linestyle="--", linewidth=0.5)

        ax2 = ax1.twinx()
        if value_loss_history:
            ax2.plot(episodes_logged[:len(value_loss_history)], value_loss_history, label="value_loss", color="tab:green")
            ax2.set_ylabel("Value loss")

        lines_labels = ax1.get_legend_handles_labels()
        if value_loss_history:
            hl2 = ax2.get_legend_handles_labels()
            lines = lines_labels[0] + hl2[0]
            labels = lines_labels[1] + hl2[1]
        else:
            lines, labels = lines_labels
        ax1.legend(lines, labels, loc="upper right")
        plt.title("PPO Diagnostics")
        plt.tight_layout()
        plt.savefig(args.ppo_diag_plot)
        plt.close()
        print(f"[RL] PPO diagnostics saved => {args.ppo_diag_plot}")

    if best_return == float("-inf"):
        print("[RL] Warning: no episodes completed; no checkpoint saved.")
    else:
        print(f"[RL] best return={best_return:.1f} checkpoint => {args.save_best}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
