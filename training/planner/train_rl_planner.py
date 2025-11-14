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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter

plt.switch_backend("Agg")

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.controller import RuleBasedController

from models.planner_model.model import DVRPNet, prepare_features, prepare_agents


class ValueCritic(nn.Module):
    """
    A small value-function critic used by PPO.

    The critic expects an input embedding produced by `aggregate_state_embedding`,
    which concatenates the depot embedding and the mean node embedding. The
    critic outputs a single scalar value per batch element.

    Args:
        d_model: the embedding dimensionality (per-component) produced by the
            planner model. The critic's input dimension is `d_model * 2` because
            the state embedding concatenates depot and mean node embeddings.
    """
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
    """
    Produce a compact state embedding used by the value critic.

    The function concatenates the depot embedding with the mean embedding of
    the (unmasked) nodes to produce a single vector per batch element.

    Args:
        enc_nodes: [B, N, D] tensor of node embeddings from the encoder.
        enc_depot: [B, 1, D] tensor of depot embeddings from the encoder.
        node_mask: [B, N] boolean mask where True indicates masked/invalid nodes.

    Returns:
        Tensor of shape [B, 2*D] containing the concatenation of depot_embed
        and node_mean. If N == 0, node_mean is a zero tensor of shape [B, D].
    """
    depot_embed = enc_depot.squeeze(1)
    if enc_nodes.size(1) == 0:
        node_mean = torch.zeros_like(depot_embed)
    else:
        valid = (~node_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        node_mean = (enc_nodes * valid).sum(dim=1) / denom
    return torch.cat([depot_embed, node_mean], dim=-1)


def detach_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Detach feature tensors from the current computation graph and move them to
    CPU. A clone is returned to ensure the saved features are independent of
    the original tensors.

    This is used when storing rollout data for later re-evaluation during PPO
    updates to avoid holding GPU memory / computation graph references.

    Args:
        feats: a dict of feature tensors (typically produced by
            `prepare_features`).

    Returns:
        A dict with the same keys where each tensor is detached, moved to CPU,
        and cloned.
    """
    return {k: v.detach().cpu().clone() for k, v in feats.items()}


def compute_returns(rewards: List[float], dones: List[bool], gamma: float, device: torch.device) -> torch.Tensor:
    """
    Compute discounted returns for a sequence of rewards.

    The implementation iterates the rewards/dones sequence in reverse and
    applies the standard discounted-return recursion:

        R_t = r_t + gamma * R_{t+1}

    When `done` is True at a step, the return accumulator R is reset to 0
    before processing that step (so episodes are handled correctly).

    Args:
        rewards: list of scalar rewards (floats) collected during the episode.
        dones: list of booleans indicating terminal steps aligned with rewards.
        gamma: discount factor in [0,1].
        device: torch device where the returned tensor should be placed.

    Returns:
        A 1-D tensor of discounted returns of shape [T] on the requested device.
    """
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
    """
    Re-evaluate a single saved decision-step sample under the current policy.

    This function is used by PPO during updates: decision-step samples are
    stored on CPU during the rollout (detached via `detach_feats`). During the
    PPO mini-batch evaluation we move these tensors back to the training
    device, run the encoder+decoder to obtain the current policy logits, and
    compute the log-probability and entropy of the originally sampled action
    under the current policy. The critic is also evaluated on the current
    policy's state embedding.

    Args:
        model: DVRPNet policy model (in train/eval mode as appropriate).
        critic: ValueCritic used to compute state values.
        sample: a dict produced by `decision_steps` containing keys
            'feats', 'agents', 'actions', etc. Note: these tensors are on CPU
            and detached from the graph.
        lateness_lambda: a float passed to model.decode (soft lateness penalty).
        device: the device to move tensors to for evaluation.

    Returns:
        A tuple (log_probs, entropies, values) where:
          - log_probs: [B] tensor of summed log-probabilities (sum over agents)
          - entropies: [B] tensor of summed entropies (sum over agents)
          - values: [B] tensor of critic values for the state
    """
    feats_cpu = sample["feats"]  # type: ignore[index]
    agents_cpu = sample["agents"]  # type: ignore[index]
    actions_cpu = sample["actions"]  # type: ignore[index]
    hist_pos_cpu = sample.get("history_positions", None)  # type: ignore[index]
    hist_idx_cpu = sample.get("history_indices", None)    # type: ignore[index]

    feats = {k: v.to(device) for k, v in feats_cpu.items()}
    agents = agents_cpu.to(device)
    actions = actions_cpu.to(device)
    history_positions = hist_pos_cpu.to(device) if hist_pos_cpu is not None else None
    history_indices = hist_idx_cpu.to(device) if hist_idx_cpu is not None else None

    if feats["nodes"].size(1) == 0:
        B = actions.size(0)
        log_prob = torch.zeros(B, device=device)
        entropy = torch.zeros(B, device=device)
        value = critic(torch.zeros(B, critic.input_dim, device=device))
        return log_prob.squeeze(-1), entropy.squeeze(-1), value.squeeze(-1)

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

    # compute per-agent log-prob and entropy using logp/probs so autograd
    # remains connected to the model logits
    log_terms = []
    ent_terms = []
    for b in range(B):
        lp = []
        ent = []
        for a in range(A):
            act = actions[b, a]
            # gather log-prob for the taken action (preserves grad)
            lp.append(logp[b, a, act])
            # entropy for this agent: -sum(p * log p)
            ent.append((-(probs[b, a] * logp[b, a]).sum()))
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
    """
    Perform PPO-style updates using stored decision-step samples.

    The function expects `decision_steps` collected during a single episode
    rollout. Each entry in `decision_steps` must include the original
    'old_log_prob' and 'value' computed at sampling time, as well as detached
    copies of 'feats', 'agents' and 'actions' to re-evaluate under the current
    policy. The function will run for `args.ppo_epochs` epochs and use
    mini-batches of size `args.ppo_batch_size` to compute the clipped surrogate
    objective, value loss and entropy bonus. Both policy and value optimizers
    are stepped inside.

    Args:
        model: policy network (DVRPNet).
        critic: value network.
        opt_policy: optimizer for the policy parameters.
        opt_value: optimizer for the critic parameters.
        decision_steps: list of saved decision-step dictionaries from rollout.
        returns_all: 1-D tensor of discounted returns aligned with time steps.
        args: parsed command-line arguments containing PPO hyperparameters.
        device: device to place temporary tensors on.
        lateness_lambda: forwarded to model.decode during evaluation.

    Returns:
        A dict with averaged metrics: policy_loss, value_loss, entropy.
    """
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
        max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
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
    p.add_argument("--ckpt_init", type=str, default="checkpoints/planner/planner_20_2_200.pt", help="Initial planner checkpoint to warm start")
    p.add_argument("--save_best", type=str, default="checkpoints/planner/planner_rl_best.pt", help="Path to save the best-performing RL checkpoint")
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


def select_targets_with_sampling(model: DVRPNet,
                                 feats: dict,
                                 agents_tensor: torch.Tensor,
                                 lateness_lambda: float,
                                 cap_full: torch.Tensor,
                                 critic: ValueCritic | None = None,
                                 history_positions: torch.Tensor | None = None,
                                 history_indices: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Sample one-step target indices for each agent according to the current
    policy probabilities produced by `model.decode`.

    The function returns sampled selection indices (`sel`), the corresponding
    destination coordinates (`dest_xy`), the log-probabilities of the sampled
    indices (`log_probs`) and, if a critic is provided, the critic value for
    the current state embedding.

    Args:
        model: DVRPNet policy model.
        feats: feature dictionary (as produced by `prepare_features`).
        agents_tensor: [B,A,...] tensor describing agent states on the chosen device.
        lateness_lambda: float penalty forwarded to the decoder.
        cap_full: tensor indicating capacities (not used directly here but kept for API compatibility).
        critic: optional ValueCritic; if provided, returns value estimates.

    Returns:
        sel: [B,A] long tensor of sampled indices (0 means depot, 1..N map to nodes).
        dest_xy: [B,A,2] long tensor of destination coordinates for each agent.
        log_probs: [B,A] float tensor of log-probabilities of sampled indices.
        value: optional [B] tensor of critic values (or None if critic is None).
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
    )  # [B,A,N+1]

    probs = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    B, A, N1 = probs.shape
    N = N1 - 1
    node_xy = feats["nodes"][..., :2].long()        # [B,N,2]
    depot_xy = feats["depot"][..., :2].long().squeeze(1)  # [B,2]

    sel = torch.zeros(B, A, dtype=torch.long, device=dev)
    dest_xy = torch.zeros(B, A, 2, dtype=torch.long, device=dev)
    log_probs = torch.zeros(B, A, dtype=torch.float32, device=dev)

    for b in range(B):
        for a in range(A):
            # print(f"Debug: B={b}, A={a}, Probs={probs[b,a].cpu().detach().numpy()}") if os.getenv("DVRP_DEBUG_LOGP") else None
            # sample using the categorical distribution built from probs
            cat = Categorical(probs[b, a])
            idx = cat.sample()  # 0..N
            sel[b, a] = idx
            # use log_softmax gather to keep autograd connected to logits
            log_probs[b, a] = logp[b, a, idx]
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
                csv_writer = csv.writer(fh)
                csv_writer.writerow(["episode", "return"])

    if args.reward_plot:
        reward_plot_dir = os.path.dirname(args.reward_plot) or "."
        os.makedirs(reward_plot_dir, exist_ok=True)

    if args.save_best:
        save_best_dir = os.path.dirname(args.save_best) or "."
        os.makedirs(save_best_dir, exist_ok=True)

    critic = None
    value_opt = None
    baseline = None
    writer: SummaryWriter | None = None
    if getattr(args, "tb_logdir", None):
        if len(args.tb_logdir.strip()) > 0:
            os.makedirs(args.tb_logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=args.tb_logdir)
    if args.algo == "ppo":
        critic = ValueCritic(model.d_model).to(device)
        critic.train()
        value_opt = torch.optim.AdamW(critic.parameters(), lr=args.value_lr)

    gamma = args.gamma

    for ep in range(1, args.episodes + 1):
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
        logprob_traj: List[torch.Tensor] = []
        decision_steps: List[Dict[str, object]] = []
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

            sel, dest_xy, log_probs, state_value = select_targets_with_sampling(
                model,
                feats,
                agents_t,
                lateness_lambda=args.lateness_lambda,
                cap_full=torch.full((1, cfg.num_agents), float(cfg.capacity), device=device),
                critic=critic,
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
            logprob_traj.append(log_prob_sum)
            rewards_all.append(reward_val)
            dones_all.append(done)
            # depot ratio 统计
            depot_select_count += int((sel == 0).sum().item())
            total_select_count += sel.numel()

            if critic is not None and state_value is not None:
                decision_steps.append({
                    "step_index": len(rewards_all) - 1,
                    "feats": detach_feats(feats),
                    "agents": agents_t.detach().cpu().clone(),
                    "actions": sel.detach().cpu().clone(),
                    "old_log_prob": float(log_prob_sum.detach().cpu().item()),
                    "value": float(state_value.detach().cpu().item()),
                    "history_positions": hp.detach().cpu().clone(),
                    "history_indices": hi.detach().cpu().clone(),
                })

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

        if args.algo == "reinforce":
            episode_return = total_reward
            if baseline is None:
                baseline = episode_return
            adv = episode_return - baseline
            baseline = 0.9 * baseline + 0.1 * episode_return

            # If no policy decisions were made (e.g., no demands), skip update safely
            if not logprob_traj:
                print(f"[EP {ep:04d}] return={total_reward:.5f} adv={adv:.1f} (skip update: no decisions)")
            else:
                sum_logprob = torch.stack(logprob_traj).sum()
                if not sum_logprob.requires_grad:
                    print(f"[EP {ep:04d}] return={total_reward:.5f} adv={adv:.1f} (skip update: no grad)")
                else:
                    loss = -adv * sum_logprob
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    print(f"[EP {ep:04d}] return={total_reward:.5f} adv={adv:.1f} loss={float(loss.item()):.3f}")
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
                f"[EP {ep:04d}] return={total_reward:.5f} policy_loss={stats['policy_loss']:.3f} "
                f"value_loss={stats['value_loss']:.3f} entropy={stats['entropy']:.3f}"
            )

        if total_reward > best_return:
            best_return = total_reward
            if args.save_best:
                torch.save({"model": model.state_dict(), "episode": ep, "return": total_reward}, args.save_best)
                print(f"[RL] new best checkpoint saved => {args.save_best} (return={total_reward:.5f})")

        # TensorBoard logging
        if writer is not None:
            env_stats = getattr(env, "_episode_stats", {})
            demand_count = env_stats.get("demand_count", 0)
            demand_capacity = env_stats.get("demand_capacity", 0.0)
            served_count = env_stats.get("served_count", 0)
            served_capacity = env_stats.get("served_capacity", 0.0)
            expired_capacity = env_stats.get("expired_capacity", 0.0)
            capacity_reward_term = env_stats.get("capacity_reward_term", 0.0)
            expired_penalty_mag = env_stats.get("expired_penalty", 0.0)
            switch_penalty_term = env_stats.get("switch_penalty", 0.0)
            exploration_penalty_value = env_stats.get("exploration_penalty_value", 0.0)
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
