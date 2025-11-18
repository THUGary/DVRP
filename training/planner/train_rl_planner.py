from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple
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
)
from training.planner.rl_algorithms.sampling import select_targets_with_sampling, detach_feats


ALGORITHM_REGISTRY = {
    "reinforce": ReinforceAlgorithm,
    "ppo": PPOAlgorithm,
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
    p.add_argument("--algo", type=str, default="reinforce", choices=["reinforce", "ppo"], help="Policy gradient algorithm to use")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns")
    p.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per episode")
    p.add_argument("--ppo_batch_size", type=int, default=32, help="Mini-batch size for PPO updates")
    p.add_argument("--ppo_clip", type=float, default=0.2, help="Clipping epsilon for PPO objective")
    p.add_argument("--value_lr", type=float, default=1e-4, help="Learning rate for the value critic (PPO)")
    p.add_argument("--value_coef", type=float, default=0.5, help="Weight for value loss in PPO")
    p.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient for PPO")
    p.add_argument("--normalize_adv", action="store_true", help="Normalize advantages before PPO update")
    p.add_argument("--debug", action="store_true", help="Enable per-step debug printing of observed demands and new demands")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic torch/CuDNN behavior (may reduce performance)")
    p.add_argument("--tb_logdir", type=str, default="runs/tb", help="TensorBoard log directory (empty string to disable)")
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

    writer: SummaryWriter | None = None
    if getattr(args, "tb_logdir", None):
        if len(args.tb_logdir.strip()) > 0:
            os.makedirs(args.tb_logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=args.tb_logdir)
    rl_algo = build_algorithm(args.algo, model, opt, device, args)

    for ep in range(1, args.episodes + 1):
        rl_algo.begin_episode(ep)
        # Reseed per-episode so that all sources of randomness (torch,
        # numpy, python random) are aligned to the episode seed. We still
        # pass the same seed to the environment generator for completeness.
        seed_ep = int(args.seed + ep)
        random.seed(seed_ep)
        np.random.seed(seed_ep)
        torch.manual_seed(seed_ep)
        if device.type == "cuda":
            try:
                torch.cuda.manual_seed_all(seed_ep)
            except Exception:
                pass

        obs = env.reset(seed=seed_ep)
        total_reward = 0.0
        rewards_all: List[float] = []
        dones_all: List[bool] = []

        # keep track of previous demands to compute "new" arrivals for debug
        prev_demands: List[Tuple[int, int, int, int, int]] = []

        # 初始化历史：记录初始位置
        done = False
        hist_pos: List[List[Tuple[int, int]]] = []  # 每个 agent 的 (x,y) 列表
        hist_idx: List[List[int]] = []              # 每个 agent 的 选择索引序列 (0=depot,1..N=node)
        for (x, y, s) in obs["agent_states"]:
            hist_pos.append([(int(x), int(y))])
            hist_idx.append([0])  # 初始位置视作 depot
        depot_select_count = 0
        total_select_count = 0
        while not done:
            nodes_list = obs["demands"]
            step_idx = len(rewards_all)
            # compute newly observed demands (those not seen in previous step)
            try:
                new_demands = [d for d in nodes_list if d not in prev_demands]
            except Exception:
                new_demands = list(nodes_list)
            if args.debug:
                if nodes_list:
                    print(f"[EP {ep:04d} STEP {step_idx}] demands={len(nodes_list)} new={len(new_demands)}")
                else:
                    print(f"[EP {ep:04d} STEP {step_idx}] no demands seen")
            N = len(nodes_list)
            node_mask = [False] * N
            depot = [(*obs["depot"], obs["time"])]

            if N == 0:
                actions = [(0, 0) for _ in obs["agent_states"]]
                next_obs, reward, done, _ = env.step(actions)
                reward_val = float(reward)
                total_reward += reward_val
                # No decision taken this step -> no policy log-prob to accumulate
                rewards_all.append(reward_val)
                dones_all.append(done)
                # update prev_demands before moving to next observation
                prev_demands = list(nodes_list)
                obs = next_obs
                continue

            feats = prepare_features(nodes=[nodes_list], node_mask=[node_mask], depot=[depot], d_model=model.d_model, device=device)
            agents = [(x, y, s, obs["time"]) for (x, y, s) in obs["agent_states"]]
            agents_t = prepare_agents([agents], device=device)

            # 组织历史位置序列 [B=1, A, T, 2]，无 padding（T 为各 agent 相同）
            T = max(len(h) for h in hist_pos)
            A = len(hist_pos)
            hp = torch.full((1, A, T, 2), -1, dtype=torch.float32, device=device)
            hi = torch.full((1, A, T), -1, dtype=torch.long, device=device)
            for a_idx, (seq_pos, seq_idx) in enumerate(zip(hist_pos, hist_idx)):
                for t_idx, (px, py) in enumerate(seq_pos):
                    hp[0, a_idx, t_idx, 0] = float(px)
                    hp[0, a_idx, t_idx, 1] = float(py)
                # 索引序列长度与位置序列一致（决策后追加），截断或填充
                for t_idx, idx_val in enumerate(seq_idx):
                    if t_idx < T:
                        hi[0, a_idx, t_idx] = int(idx_val)

            critic_module = getattr(rl_algo, "critic", None)
            sel, dest_xy, log_probs, state_value = select_targets_with_sampling(
                model=model,
                feats=feats,
                agents_tensor=agents_t,
                lateness_lambda=args.lateness_lambda,
                critic=critic_module,
                history_positions=hp,
                history_indices=hi,
            )

            if args.debug:
                # 统计本步各 agent 选择 depot 的比例
                B, A = sel.shape
                depot_sel = int((sel == 0).sum().item())
                total = B * A
                print(f"[EP {ep:04d} STEP {step_idx}] depot_ratio={depot_sel}/{total} = {depot_sel/float(max(1,total)):.2f}")

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
            # depot ratio 统计
            depot_select_count += int((sel == 0).sum().item())
            total_select_count += sel.numel()

            record = DecisionRecord(
                step_index=len(rewards_all) - 1,
                log_prob_sum=log_prob_sum,
            )
            if rl_algo.requires_full_state:
                record.feats = detach_feats(feats)
                record.agents = agents_t.detach().cpu().clone()
                record.actions = sel.detach().cpu().clone()
                record.state_value = state_value.detach().cpu().clone() if state_value is not None else None
                record.history_positions = hp.detach().cpu().clone()
                record.history_indices = hi.detach().cpu().clone()
            rl_algo.record_decision(record)

            obs = next_obs
            # remember demands seen at this step for the next iteration's diff
            prev_demands = list(nodes_list)
            # 更新历史：追加新位置（下一状态）
            # 更新历史：追加下一状态位置与本步选择的索引（sel 已对应目标点，长度与 agent 数一致）
            hist_pos = [seq + [(int(x), int(y))] for seq, (x, y, s) in zip(hist_pos, obs["agent_states"]) ]
            hist_idx = [seq + [int(sel[0, a].item())] for a, seq in enumerate(hist_idx)]

        reward_history.append(total_reward)
        if args.reward_log:
            with open(args.reward_log, "a", newline="") as fh:
                csv_writer = csv.writer(fh)
                csv_writer.writerow([ep, total_reward])

        env_stats = getattr(env, "_episode_stats", {})
        stats = rl_algo.end_episode(total_reward, rewards_all, dones_all, env_stats)
        msg = format_metrics(stats)
        print(f"[EP {ep:04d}] return={total_reward:.5f}{msg}")

        if total_reward > best_return:
            best_return = total_reward
            if args.save_best:
                torch.save({"model": model.state_dict(), "episode": ep, "return": total_reward}, args.save_best)
                print(f"[RL] new best checkpoint saved => {args.save_best} (return={total_reward:.5f})")

        # TensorBoard logging
        if writer is not None:
            demand_count = env_stats.get("demand_count", 0)
            demand_capacity = env_stats.get("demand_capacity", 0.0)
            served_count = env_stats.get("served_count", 0)
            served_capacity = env_stats.get("served_capacity", 0.0)
            expired_capacity = env_stats.get("expired_capacity", 0.0)
            capacity_reward_term = env_stats.get("capacity_reward_term", 0.0)
            expired_penalty_mag = env_stats.get("expired_penalty", 0.0)
            switch_penalty_term = env_stats.get("switch_penalty", 0.0)
            exploration_penalty_value = env_stats.get("exploration_penalty_value", 0.0)
            pairwise_penalty_value = env_stats.get("pairwise_penalty_value", 0.0)
            served_ratio = (served_capacity / demand_capacity) if demand_capacity > 1e-9 else 0.0
            depot_ratio = (depot_select_count / total_select_count) if total_select_count > 0 else 0.0
            # expiry penalty sign restore (original per-step negative)
            expiry_penalty_total = -expired_penalty_mag
            writer.add_scalar("episode/return", total_reward, ep)
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

    if best_return == float("-inf"):
        print("[RL] Warning: no episodes completed; no checkpoint saved.")
    else:
        print(f"[RL] best return={best_return:.1f} checkpoint => {args.save_best}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
