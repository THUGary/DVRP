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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque

plt.switch_backend("Agg")

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.controller import RuleBasedController

from models.planner_model.model import DVRPNet, prepare_features, prepare_agents


class ValueCritic(nn.Module):
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


def aggregate_state_embedding(enc_nodes: torch.Tensor, enc_depot: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    depot_embed = enc_depot.squeeze(1)
    if enc_nodes.size(1) == 0:
        node_mean = torch.zeros_like(depot_embed)
    else:
        valid = (~node_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        node_mean = (enc_nodes * valid).sum(dim=1) / denom
    return torch.cat([depot_embed, node_mean], dim=-1)


def detach_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in feats.items()}


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


def evaluate_sample(model: DVRPNet,
                    critic: ValueCritic,
                    sample: Dict[str, object],
                    lateness_lambda: float,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats_cpu = sample["feats"]  # type: ignore[index]
    agents_cpu = sample["agents"]  # type: ignore[index]
    actions_cpu = sample["actions"]  # type: ignore[index]

    feats = {k: v.to(device) for k, v in feats_cpu.items()}
    agents = agents_cpu.to(device)
    actions = actions_cpu.to(device)

    if feats["nodes"].size(1) == 0:
        B = actions.size(0)
        log_prob = torch.zeros(B, device=device)
        entropy = torch.zeros(B, device=device)
        value = critic(torch.zeros(B, critic.input_dim, device=device))
        return log_prob.squeeze(-1), entropy.squeeze(-1), value.squeeze(-1)

    enc_nodes, enc_depot, node_mask = model.encoder(feats)
    logits = model.decode(
        enc_nodes=enc_nodes,
        enc_depot=enc_depot,
        node_mask=node_mask,
        agents_tensor=agents,
        nodes=feats.get("nodes"),
        lateness_lambda=lateness_lambda,
    )
    probs = torch.softmax(logits, dim=-1)
    B, A, _ = probs.shape

    log_terms = []
    ent_terms = []
    for b in range(B):
        lp = []
        ent = []
        for a in range(A):
            cat = Categorical(probs[b, a])
            act = actions[b, a]
            lp.append(cat.log_prob(act))
            ent.append(cat.entropy())
        log_terms.append(torch.stack(lp).sum())
        ent_terms.append(torch.stack(ent).sum())

    state_embed = aggregate_state_embedding(enc_nodes, enc_depot, node_mask)
    value = critic(state_embed)
    return torch.stack(log_terms), torch.stack(ent_terms), value.squeeze(-1)


def ppo_update(model: DVRPNet,
               critic: ValueCritic,
               opt_policy: torch.optim.Optimizer,
               opt_value: torch.optim.Optimizer,
               decision_steps: List[Dict[str, object]],
               returns_all: torch.Tensor,
               args: argparse.Namespace,
               device: torch.device,
               lateness_lambda: float) -> Dict[str, float]:
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


def build_env_from_cfg(cfg: Config) -> GridEnvironment:
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
    )
    env.num_agents = cfg.num_agents
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL fine-tuning for DVRPNet (policy gradient)")
    p.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--ckpt_init", type=str, default="checkpoints/planner/planner_20_2_200.pt", help="Initial planner checkpoint to warm start")
    p.add_argument("--save_best", type=str, default="checkpoints/planner/planner_rl_best.pt", help="Path to save the best-performing RL checkpoint")
    p.add_argument("--generator", type=str, choices=["rule", "net"], default=None, help="Override generator type for RL training")
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
    return p.parse_args()


def select_targets_with_sampling(model: DVRPNet,
                                 feats: dict,
                                 agents_tensor: torch.Tensor,
                                 lateness_lambda: float,
                                 cap_full: torch.Tensor,
                                 critic: ValueCritic | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Sample a one-step target for each agent using softmax over logits.
    Returns (sel_indices [B,A], dest_xy [B,A,2], log_probs [B,A]).
    """
    dev = agents_tensor.device
    if feats["nodes"].size(1) == 0:
        B = agents_tensor.size(0)
        A = agents_tensor.size(1)
        depot_xy = feats["depot"][..., :2].long().squeeze(1)  # [B,2]
        sel = torch.zeros(B, A, dtype=torch.long, device=dev)
        dest_xy = depot_xy.unsqueeze(1).repeat(1, A, 1)
        log_probs = torch.zeros(B, A, dtype=torch.float32, device=dev)
        value = None
        if critic is not None:
            embed = torch.zeros(B, critic.input_dim, device=dev)
            value = critic(embed)
        return sel, dest_xy, log_probs, value

    enc_nodes, enc_depot, node_mask = model.encoder(feats)
    logits = model.decode(
        enc_nodes=enc_nodes,
        enc_depot=enc_depot,
        node_mask=node_mask,
        agents_tensor=agents_tensor,
        nodes=feats["nodes"],
        lateness_lambda=lateness_lambda,
    )  # [B,A,N+1]

    probs = torch.softmax(logits, dim=-1)
    B, A, N1 = probs.shape
    N = N1 - 1
    node_xy = feats["nodes"][..., :2].long()        # [B,N,2]
    depot_xy = feats["depot"][..., :2].long().squeeze(1)  # [B,2]

    sel = torch.zeros(B, A, dtype=torch.long, device=dev)
    dest_xy = torch.zeros(B, A, 2, dtype=torch.long, device=dev)
    log_probs = torch.zeros(B, A, dtype=torch.float32, device=dev)

    for b in range(B):
        for a in range(A):
            cat = Categorical(probs[b, a])
            idx = cat.sample()  # 0..N
            sel[b, a] = idx
            log_probs[b, a] = cat.log_prob(idx)
            if 1 <= idx <= N:
                dest_xy[b, a] = node_xy[b, idx - 1]
            else:
                dest_xy[b, a] = depot_xy[b]

    value = None
    if critic is not None:
        state_embed = aggregate_state_embedding(enc_nodes, enc_depot, node_mask)
        value = critic(state_embed)

    return sel, dest_xy, log_probs, value


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

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

    if args.ckpt_init and os.path.exists(args.ckpt_init):
        blob = torch.load(args.ckpt_init, map_location=device)
        state = blob.get("model", blob)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[RL] Warm start from {args.ckpt_init} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[RL] No warm-start checkpoint found; training from random init.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    reward_history: List[float] = []
    best_return = float("-inf")

    if args.reward_log:
        reward_log_dir = os.path.dirname(args.reward_log) or "."
        os.makedirs(reward_log_dir, exist_ok=True)
        if not os.path.exists(args.reward_log):
            with open(args.reward_log, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["episode", "return"])

    if args.reward_plot:
        reward_plot_dir = os.path.dirname(args.reward_plot) or "."
        os.makedirs(reward_plot_dir, exist_ok=True)

    if args.save_best:
        save_best_dir = os.path.dirname(args.save_best) or "."
        os.makedirs(save_best_dir, exist_ok=True)

    critic = None
    value_opt = None
    baseline = None
    if args.algo == "ppo":
        critic = ValueCritic(model.d_model).to(device)
        critic.train()
        value_opt = torch.optim.AdamW(critic.parameters(), lr=args.value_lr)

    gamma = args.gamma

    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        logprob_traj: List[torch.Tensor] = []
        decision_steps: List[Dict[str, object]] = []
        rewards_all: List[float] = []
        dones_all: List[bool] = []

        done = False
        while not done:
            nodes_list = obs["demands"]
            N = len(nodes_list)
            node_mask = [False] * N
            depot = [(*obs["depot"], obs["time"])]

            if N == 0:
                actions = [(0, 0) for _ in obs["agent_states"]]
                next_obs, reward, done, _ = env.step(actions)
                reward_val = float(reward)
                total_reward += reward_val
                logprob_traj.append(torch.tensor(0.0, device=device))
                rewards_all.append(reward_val)
                dones_all.append(done)
                obs = next_obs
                continue

            feats = prepare_features(nodes=[nodes_list], node_mask=[node_mask], depot=[depot], d_model=model.d_model, device=device)
            agents = [(x, y, s, obs["time"]) for (x, y, s) in obs["agent_states"]]
            agents_t = prepare_agents([agents], device=device)

            sel, dest_xy, log_probs, state_value = select_targets_with_sampling(
                model,
                feats,
                agents_t,
                lateness_lambda=args.lateness_lambda,
                cap_full=torch.full((1, cfg.num_agents), float(cfg.capacity), device=device),
                critic=critic,
            )

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
            logprob_traj.append(log_prob_sum)
            rewards_all.append(reward_val)
            dones_all.append(done)

            if critic is not None and state_value is not None:
                decision_steps.append({
                    "step_index": len(rewards_all) - 1,
                    "feats": detach_feats(feats),
                    "agents": agents_t.detach().cpu().clone(),
                    "actions": sel.detach().cpu().clone(),
                    "old_log_prob": float(log_prob_sum.detach().cpu().item()),
                    "value": float(state_value.detach().cpu().item()),
                })

            obs = next_obs

        reward_history.append(total_reward)
        if args.reward_log:
            with open(args.reward_log, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([ep, total_reward])

        if args.algo == "reinforce":
            episode_return = total_reward
            if baseline is None:
                baseline = episode_return
            adv = episode_return - baseline
            baseline = 0.9 * baseline + 0.1 * episode_return

            if not logprob_traj:
                logprob_traj.append(torch.tensor(0.0, device=device))
            sum_logprob = torch.stack(logprob_traj).sum()
            loss = -adv * sum_logprob

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            print(f"[EP {ep:04d}] return={total_reward:.1f} adv={adv:.1f} loss={float(loss.item()):.3f}")
        else:
            assert critic is not None and value_opt is not None
            if not rewards_all:
                rewards_all.append(0.0)
                dones_all.append(True)
            returns_all = compute_returns(rewards_all, dones_all, gamma, device)
            stats = ppo_update(
                model,
                critic,
                opt,
                value_opt,
                decision_steps,
                returns_all,
                args,
                device,
                args.lateness_lambda,
            )
            print(
                f"[EP {ep:04d}] return={total_reward:.1f} policy_loss={stats['policy_loss']:.3f} "
                f"value_loss={stats['value_loss']:.3f} entropy={stats['entropy']:.3f}"
            )

        if total_reward > best_return:
            best_return = total_reward
            if args.save_best:
                torch.save({"model": model.state_dict(), "episode": ep, "return": total_reward}, args.save_best)
                print(f"[RL] new best checkpoint saved => {args.save_best} (return={total_reward:.1f})")

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


if __name__ == "__main__":
    main()
