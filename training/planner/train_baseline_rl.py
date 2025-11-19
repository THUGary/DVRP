from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import datetime

# Ensure project root on sys.path
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.controller import RuleBasedController
from models.planner_model.model import DVRPNet, prepare_features, prepare_agents
from training.planner.rl_algorithms.sampling import select_targets_with_sampling


def set_global_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


@dataclass
class EpisodeResult:
    total_reward: float
    log_probs: List[torch.Tensor]
    entropies: List[torch.Tensor]
    rewards: List[float]
    dones: List[bool]
    env_stats: Dict[str, float]


def build_env_from_cfg(cfg: Config) -> GridEnvironment:
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
    p = argparse.ArgumentParser(description="REINFORCE training with explicit baseline model comparison")
    p.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the policy model")
    p.add_argument("--seed", type=int, default=0, help="Global random seed")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for training")
    p.add_argument("--ckpt_init", type=str, default=None, help="Optional checkpoint to warm start the policy model")
    p.add_argument("--baseline_ckpt", type=str, default=None, help="Optional checkpoint to warm start the baseline model")
    p.add_argument(
        "--save_best",
        type=str,
        default="checkpoints/planner_rl",
        help="Base filename (without episode suffix) used when storing periodic checkpoints",
    )
    p.add_argument("--generator", type=str, choices=["rule", "net"], default="rule", help="Demand generator override")
    p.add_argument("--lateness_lambda", type=float, default=0.0, help="Soft lateness penalty during decode")
    p.add_argument("--target_queue_len", type=int, default=1, help="Length of autoregressive queue during rollout")
    p.add_argument("--reward_log", type=str, default="runs/baseline_rl_rewards.csv", help="CSV log for episode rewards")
    p.add_argument("--update_cycle", type=int, default=20, help="Number of training episodes between baseline evaluation cycles")
    p.add_argument("--val_num", type=int, default=15, help="Number of validation environments in the evaluation dataset")
    p.add_argument("--val_data_path", type=str, default="training/planner/data/baseline_val.pt", help="Path to the validation seed dataset file")
    p.add_argument("--gen_val_data", action="store_true", help="Regenerate the validation seed dataset before training")
    p.add_argument("--entropy_coef", type=float, default=0.01, help="Initial entropy regularization coefficient λ")
    p.add_argument(
        "--entropy_decay",
        type=float,
        default=1.0,
        help="Multiplicative decay applied to λ after each episode (use <1.0 to decay)",
    )
    p.add_argument("--debug", action="store_true", help="Print per-step demand information")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic CuDNN behavior")
    p.add_argument("--run_name", type=str, default=None, help="Name for the run (used in TensorBoard and plot filenames)")
    return p.parse_args()


def plot_training_curves(csv_path: str, output_dir: str, run_name: str):
    """
    Reads the reward CSV log and generates static plots for Reward, Advantage, etc.
    """
    if not os.path.exists(csv_path):
        return

    episodes = []
    policy_rewards = []
    baseline_rewards = []
    advantages = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                policy_rewards.append(float(row["policy_reward"]))
                baseline_rewards.append(float(row["baseline_reward"]))
                advantages.append(float(row["advantage"]))
    except Exception as e:
        print(f"[PLOT] Error reading CSV: {e}")
        return

    if not episodes:
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot 1: Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, policy_rewards, label="Policy Reward", alpha=0.7)
    plt.plot(episodes, baseline_rewards, label="Baseline Reward", alpha=0.7, linestyle="--")
    
    # Calculate moving average for policy reward
    window_size = min(50, len(policy_rewards))
    if window_size > 1:
        ma = np.convolve(policy_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(episodes[window_size-1:], ma, label=f"Policy MA({window_size})", color='red', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Training Rewards - {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{run_name}_rewards_{timestamp}.png"))
    plt.close()

    # Plot 2: Advantage
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, advantages, label="Advantage", color='purple', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Advantage (Policy - Baseline)")
    plt.title(f"Advantage over Baseline - {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{run_name}_advantage_{timestamp}.png"))
    plt.close()
    
    print(f"[PLOT] Saved training plots to {output_dir}")


def ensure_log_file(path: str, headers: List[str]) -> None:
    if not path:
        return
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(headers)


def _resolve_path(path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str)
    if not p.is_absolute():
        p = _ROOT / p
    return p


def _checkpoint_dir_and_prefix(args: argparse.Namespace) -> Tuple[pathlib.Path, str]:
    base_dir = _ROOT / "checkpoints" / "planner"
    base_dir.mkdir(parents=True, exist_ok=True)
    template = pathlib.Path(args.save_best)
    prefix = template.stem if template.stem else "planner_rl"
    return base_dir, prefix


def _generate_val_seeds(seed: int, count: int) -> List[int]:
    rng = random.Random(seed)
    max_int = 2 ** 31 - 1
    return [rng.randint(0, max_int) for _ in range(count)]


def _save_val_dataset(path: pathlib.Path, seeds: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"seeds": seeds}, fh)


def _load_val_dataset(path: pathlib.Path) -> List[int]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "seeds" in data:
        data = data["seeds"]
    if not isinstance(data, list):
        raise ValueError(f"Validation dataset at {path} is not a list of seeds")
    if not all(isinstance(x, int) for x in data):
        raise ValueError(f"Validation dataset at {path} contains non-integer entries")
    return data


def prepare_val_seeds(args: argparse.Namespace) -> Tuple[List[int], pathlib.Path]:
    path = _resolve_path(args.val_data_path)
    seeds: List[int]
    if args.gen_val_data:
        seeds = _generate_val_seeds(args.seed + 1024, args.val_num)
        _save_val_dataset(path, seeds)
        print(f"[VAL] Generated {len(seeds)} validation seeds => {path}")
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"Validation dataset not found at {path}. Either provide --gen_val_data or ensure the file exists."
            )
        seeds = _load_val_dataset(path)
    if len(seeds) < args.val_num:
        raise ValueError(
            f"Validation dataset at {path} contains {len(seeds)} seeds, but val_num={args.val_num} was requested."
        )
    return seeds[:args.val_num], path


def run_episode(
    *,
    model: DVRPNet,
    env: GridEnvironment,
    controller: RuleBasedController,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    collect_traces: bool,
    env_verbose: bool = True,
    action_selection: str = "stochastic",
) -> EpisodeResult:
    set_global_seed(seed, device)
    ctx = torch.enable_grad() if collect_traces else torch.no_grad()
    with ctx:
        obs = env.reset(seed=seed)
        total_reward = 0.0
        rewards_all: List[float] = []
        dones_all: List[bool] = []
        log_prob_terms: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []

        prev_demands: List[Tuple[int, int, int, int, int]] = []
        hist_pos: List[List[Tuple[int, int]]] = []
        hist_targets: List[List[Tuple[int, int]]] = []
        depot_xy0 = (int(obs["depot"][0]), int(obs["depot"][1]))
        for (x, y, _s) in obs["agent_states"]:
            hist_pos.append([(int(x), int(y))])
            hist_targets.append([depot_xy0])

        done = False
        while not done:
            nodes_list = obs["demands"]
            step_idx = len(rewards_all)
            if args.debug:
                try:
                    new_demands = [d for d in nodes_list if d not in prev_demands]
                except Exception:
                    new_demands = list(nodes_list)
                if nodes_list:
                    print(f"[STEP {step_idx:03d}] demands={len(nodes_list)} new={len(new_demands)}")
                else:
                    print(f"[STEP {step_idx:03d}] no demands")

            N = len(nodes_list)
            node_mask = [False] * N
            depot = [obs["depot"]]

            if N == 0:
                actions = [(0, 0) for _ in obs["agent_states"]]
                next_obs, reward, done, _ = env.step(actions, verbose=env_verbose)
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

            sel, dest_xy, log_probs, entropies, _state_value, _queue_indices, _queue_coords = select_targets_with_sampling(
                model=model,
                feats=feats,
                agents_tensor=agents_t,
                lateness_lambda=args.lateness_lambda,
                critic=None,
                history_positions=hp,
                history_target_coords=ht,
                target_queue_len=args.target_queue_len,
                selection_strategy=action_selection,
            )

            actions: List[Tuple[int, int]] = []
            for i, (x, y, _s) in enumerate(obs["agent_states"]):
                tx, ty = int(dest_xy[0, i, 0].item()), int(dest_xy[0, i, 1].item())
                q = deque()
                q.append((tx, ty))
                actions.append(controller.act((x, y), q))

            next_obs, reward, done, _ = env.step(actions, verbose=env_verbose)
            reward_val = float(reward)
            total_reward += reward_val
            rewards_all.append(reward_val)
            dones_all.append(done)

            if collect_traces:
                log_prob_terms.append(log_probs.sum())
                entropy_terms.append(entropies.sum())

            obs = next_obs
            prev_demands = list(nodes_list)
            hist_pos = [seq + [(int(x), int(y))] for seq, (x, y, _s) in zip(hist_pos, obs["agent_states"])]
            hist_targets = [
                seq + [(int(dest_xy[0, a, 0].item()), int(dest_xy[0, a, 1].item()))]
                for a, seq in enumerate(hist_targets)
            ]

        env_stats = getattr(env, "_episode_stats", {}).copy()
        return EpisodeResult(
            total_reward=total_reward,
            log_probs=log_prob_terms,
            entropies=entropy_terms,
            rewards=rewards_all,
            dones=dones_all,
            env_stats=env_stats,
        )


def evaluate_model_on_dataset(
    *,
    model: DVRPNet,
    cfg: Config,
    args: argparse.Namespace,
    device: torch.device,
    seeds: List[int]
) -> float:
    if not seeds:
        return 0.0
    model.eval()
    env = build_env_from_cfg(cfg)
    controller = RuleBasedController(**cfg.controller_params)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.debug = False
    total = 0.0
    for seed in seeds:
        res = run_episode(
            model=model,
            env=env,
            controller=controller,
            args=eval_args,
            device=device,
            seed=seed,
            collect_traces=False,
            env_verbose=False,
            action_selection="greedy",
        )
        total += res.total_reward
    return total / float(len(seeds))


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

    env_train = build_env_from_cfg(cfg)
    env_baseline = build_env_from_cfg(cfg)
    controller_train = RuleBasedController(**cfg.controller_params)
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
    if args.baseline_ckpt:
        print("[BASELINE] --baseline_ckpt is ignored; baseline now clones the policy initialization.")
    baseline_model.load_state_dict(model.state_dict())
    print("[BASELINE] Initialized from policy parameters")
    baseline_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # TensorBoard setup
    run_name = args.run_name if args.run_name else f"planner_rl_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(_ROOT, "training/planner/runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TB] TensorBoard logging to {log_dir}")

    # Update reward log path to be inside the run directory if default
    if args.reward_log == "runs/baseline_rl_rewards.csv":
        args.reward_log = os.path.join(log_dir, "rewards.csv")
    
    ensure_log_file(args.reward_log, ["episode", "policy_reward", "baseline_reward", "advantage"])

    save_dir, save_prefix = _checkpoint_dir_and_prefix(args)

    def dump_checkpoint(ep: int) -> Optional[pathlib.Path]:
        if ep <= 0:
            return None
        path = save_dir / f"{save_prefix}_{ep}.pt"
        torch.save({"model": model.state_dict(), "episode": ep}, path)
        print(f"[SAVE] checkpoint => {path}")
        return path

    last_completed_episode = 0
    latest_checkpoint: Optional[pathlib.Path] = None

    try:
        for ep in range(1, args.episodes + 1):
            seed_ep = args.seed + ep
            model.train()
            train_result = run_episode(
                model=model,
                env=env_train,
                controller=controller_train,
                args=args,
                device=device,
                seed=seed_ep,
                collect_traces=True,
                env_verbose=True,
                action_selection="stochastic",
            )

            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.debug = False
            baseline_result = run_episode(
                model=baseline_model,
                env=env_baseline,
                controller=controller_baseline,
                args=baseline_args,
                device=device,
                seed=seed_ep,
                collect_traces=False,
                env_verbose=False,
                action_selection="greedy",
            )

            advantage = train_result.total_reward - baseline_result.total_reward
            loss_value: Optional[float] = None
            entropy_value: Optional[float] = None

            if train_result.log_probs:
                # Normalize log-prob and entropy by the number of steps (mean over time)
                # This keeps the loss magnitude stable regardless of episode length.
                logprob_stack = torch.stack(train_result.log_probs)
                sum_logprob = logprob_stack.mean()

                if train_result.entropies:
                    entropy_stack = torch.stack(train_result.entropies)
                    entropy_mean = entropy_stack.mean()
                else:
                    entropy_mean = torch.zeros(1, device=sum_logprob.device)

                current_entropy_coef = max(0.0, args.entropy_coef * (args.entropy_decay ** max(0, ep - 1)))
                loss_rl = -advantage * sum_logprob
                loss = loss_rl - current_entropy_coef * entropy_mean
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_value = float(loss.item())
                entropy_value = float((current_entropy_coef * entropy_mean).item()) if current_entropy_coef > 0 else 0.0
                
                # TensorBoard logging: Training Step
                writer.add_scalar("Train/Loss_Total", loss_value, ep)
                writer.add_scalar("Train/Loss_Policy", float(loss_rl.item()), ep)
                writer.add_scalar("Train/Entropy_Coef", current_entropy_coef, ep)
                if train_result.entropies:
                    writer.add_scalar("Train/Entropy_Raw", float(entropy_mean.item()), ep)
            else:
                print(f"[EP {ep:04d}] No actionable steps; skipping update")

            # TensorBoard logging: Episode Stats
            writer.add_scalar("Reward/Policy", train_result.total_reward, ep)
            writer.add_scalar("Reward/Baseline", baseline_result.total_reward, ep)
            writer.add_scalar("Reward/Advantage", advantage, ep)
            writer.add_scalar("Episode/Length", len(train_result.rewards), ep)
            
            # Log detailed environment stats if available
            for key, val in train_result.env_stats.items():
                if isinstance(val, (int, float)):
                    writer.add_scalar(f"EnvStats/{key}", val, ep)

            if args.reward_log:
                with open(args.reward_log, "a", newline="") as fh:
                    csv.writer(fh).writerow([ep, train_result.total_reward, baseline_result.total_reward, advantage])

            status = "encourage" if advantage >= 0 else "penalize"
            print(
                f"[EP {ep:04d}] policy={train_result.total_reward:.2f} baseline={baseline_result.total_reward:.2f} "
                f"adv={advantage:.2f} action={status}"
                + (f" loss={loss_value:.4f}" if loss_value is not None else "")
                + (
                    f" entropy_term={entropy_value:.4f}"
                    if entropy_value is not None and args.entropy_coef > 0
                    else ""
                )
            )

            if ep % 100 == 0:
                latest_checkpoint = dump_checkpoint(ep) or latest_checkpoint

            if ep % args.update_cycle == 0:
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
                print(f"[VAL][EP {ep:04d}] policy_avg={avg_policy:.2f} baseline_avg={avg_baseline:.2f}")
                
                # TensorBoard logging: Validation
                writer.add_scalar("Val/Policy_Avg_Reward", avg_policy, ep)
                writer.add_scalar("Val/Baseline_Avg_Reward", avg_baseline, ep)
                
                if avg_policy > avg_baseline:
                    baseline_model.load_state_dict(model.state_dict())
                    baseline_model.eval()
                    print(f"[BASELINE] Updated baseline model at EP {ep:04d}")
                    writer.add_scalar("Baseline/Updates", 1, ep)
                else:
                    writer.add_scalar("Baseline/Updates", 0, ep)

            last_completed_episode = ep

    except KeyboardInterrupt:
        print("[INTERRUPT] Training interrupted; final checkpoint will be saved.")
    finally:
        final_path = dump_checkpoint(last_completed_episode)
        if final_path is not None:
            latest_checkpoint = final_path
        
        # Generate static plots at the end
        if args.reward_log and os.path.exists(args.reward_log):
            plot_training_curves(args.reward_log, log_dir, run_name)
            
        writer.close()

    if latest_checkpoint is not None:
        print(f"[DONE] Latest checkpoint saved to {latest_checkpoint}")
    else:
        print("[DONE] Training finished but no episodes were completed; no checkpoint saved.")


if __name__ == "__main__":
    main()
