"""RL-style adversarial training for the diffusion demand generator to MINIMIZE a chosen planner's reward.

Goal: Learn demand distribution parameters via conditional diffusion so that a fixed planner (greedy or DVRPNet model planner)
obtains the lowest possible environment reward. We treat the generator (diffusion model) as a stochastic policy producing a set
of demands for an episode. Reward signal: negative of episode cumulative reward returned by `GridEnvironment`.

Algorithm (REINFORCE-style on diffusion):
1. Sample K episodes. For each episode:
   - Sample a latent noise z ~ N(0,1) and generate demands via diffusion conditioned on current generator params.
   - Roll out the environment with the selected planner to obtain episode reward R_env.
   - Define generator reward R_gen = - R_env.
2. For each episode, we compute standard diffusion noise-prediction loss L_diff = MSE(predicted_noise, true_noise).
3. Weight the loss by an advantage (here raw R_gen or normalized baseline-subtracted) to push distribution towards adversarial demands.
4. Update diffusion model parameters.

Simplifications:
- We treat the entire demand set generation as one action; finer-grained sequential diffusion RL is future work.
- Baseline uses exponential moving average of rewards to reduce variance.

Constraints:
- Generated demands must respect config param ranges: time in [0, max_time-1], x,y in grid bounds, capacity in [1,max_c], lifetime in [min_lifetime,max_lifetime].
  We enforce by clipping / rounding after un-normalization similarly to NetDemandGenerator.

CLI Example:
python training/generator/train_rl_diffusion_generator.py --episodes 50 --planner greedy --device cuda
python training/generator/train_rl_diffusion_generator.py --episodes 50 --planner model --planner_ckpt checkpoints/planner/planner_dynamic_20_2_200.pt --device cuda

Outputs:
- Checkpoints saved to a unique timestamped directory inside `checkpoints/rl_generator/`.
- TensorBoard logs and a CSV log saved to a unique timestamped directory inside `runs/rl_generator/`.
"""
from __future__ import annotations
import argparse
import os
import sys
import pathlib
import random
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Robust project root discovery
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import prepare_condition, CONDITION_DIM
from environment.env import GridEnvironment
from agent.planner.rule_planner import RuleBasedPlanner
from agent.planner.model_planner import ModelPlanner
from agent.planner.base import AgentState
from utils.pygame_renderer import PygameRenderer
from configs import get_default_config

# Import utils
from training.generator.rl_utils import (
    make_environment,
    apply_static_constraints,
    calculate_spatial_entropy,
    calculate_temporal_entropy,
    decode_demands_from_tensor,
    normalize_demands_for_training,
    log_density_heatmap  # [NEW] Import
)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adversarial RL training for diffusion demand generator")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--planner", type=str, default="greedy", choices=["greedy", "model"])
    p.add_argument("--planner_ckpt", type=str, default="checkpoints/planner/planner_dynamic_20_2_200.pt")
    p.add_argument("--total_demand", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init_diffusion_ckpt", type=str, default="checkpoints/diffusion_model.pth")
    p.add_argument("--log_tb", action="store_true")
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--normalize_reward", action="store_true")
    p.add_argument("--baseline_beta", type=float, default=0.9)
    p.add_argument("--sl_weight", type=float, default=0.1)
    p.add_argument("--diff_loss_clip", type=float, default=10.0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save_frames_dir", type=str, default="")
    p.add_argument("--randomize_depot", action="store_false")
    p.add_argument("--debug_planner", action="store_true")
    p.add_argument("--static", action="store_true")
    p.add_argument("--entropy_weight", type=float, default=0.80)
    p.add_argument("--time_entropy_weight", type=float, default=0.05)
    # [NEW] Control visualization frequency
    p.add_argument("--log_image_every", type=int, default=100, help="Log heatmaps every N episodes")
    # [NEW] Early Stopping
    p.add_argument("--patience", type=int, default=500, help="Stop if no improvement in generator reward for N episodes")
    return p

def _init_planner(planner_type: str, cfg, device: torch.device, ckpt_path: str | None) -> Any:
    full_cap = cfg.capacity
    if planner_type == "greedy":
        return RuleBasedPlanner(full_capacity=full_cap)
    elif planner_type == "model":
        mp = ModelPlanner(
            d_model=cfg.model_planner_params.get("d_model", 128),
            nhead=cfg.model_planner_params.get("nhead", 8),
            nlayers=cfg.model_planner_params.get("nlayers", 2),
            time_plan=cfg.model_planner_params.get("time_plan", 3),
            lateness_lambda=cfg.model_planner_params.get("lateness_lambda", 0.0),
            device=str(device),
            full_capacity=full_cap,
            coord_norm=cfg.model_planner_params.get("coord_norm", cfg.width),
            capacity_norm=cfg.model_planner_params.get("capacity_norm", cfg.capacity),
            time_norm=cfg.model_planner_params.get("time_norm", cfg.max_time),
            adapter_dim=cfg.model_planner_params.get("adapter_dim", 0),
        )
        ckpt = ckpt_path or cfg.model_planner_params.get("ckpt")
        if ckpt and os.path.exists(ckpt):
            mp.load_from_ckpt(ckpt)
            print(f"[Planner] Loaded model planner checkpoint: {ckpt}")
        else:
            print(f"[Planner] WARNING: checkpoint not found at {ckpt}; using random weights.")
        return mp
    else:
        raise ValueError(f"Unsupported planner type: {planner_type}")

def _plan_episode(planner, env: GridEnvironment, demands: List[Tuple[int,int,int,int,int]], *, renderer: PygameRenderer | None = None, fps: int = 10, save_frames_dir: str = "", debug: bool = False) -> Tuple[float, float, float]:
    """Roll out episode and return rewards."""
    obs = env.reset()
    total_initial_demand_capacity = sum(d[3] for d in demands)
    
    # Inject demands
    if hasattr(env, "_state") and env._state is not None:
        from agent.generator.base import Demand
        def _as_demand(raw: Tuple[int, ...]) -> Demand:
            service_time = int(raw[5]) if len(raw) > 5 else 0
            return Demand(x=raw[0], y=raw[1], t=raw[2], c=raw[3], end_t=raw[4], service_time=service_time)
        env._state.demands.extend([_as_demand(d) for d in demands])
        
    total_reward = 0.0
    done = False
    frame_idx = 0
    clock = None
    if renderer:
        try:
            import pygame
            clock = pygame.time.Clock()
        except Exception:
            clock = None

    while not done:
        obs_demands = obs["demands"]
        agent_states = [AgentState(x=a[0], y=a[1], s=a[2]) for a in obs["agent_states"]]
        depot = tuple(obs["depot"])
        
        plans = planner.plan(observations=obs_demands, agent_states=agent_states, depot=depot, t=obs["time"], horizon=1)
        actions = []
        
        # Simple greedy execution logic
        for a_idx, queue in enumerate(plans):
            if len(queue) == 0:
                actions.append((0,0))
            else:
                tx, ty = queue[0]
                ax, ay, _s = obs["agent_states"][a_idx]
                raw_dx, raw_dy = tx - ax, ty - ay
                step_dx, step_dy = 0, 0
                
                if raw_dx != 0 or raw_dy != 0:
                    if raw_dx != 0 and raw_dy != 0:
                        prefer_x = (abs(raw_dx) >= abs(raw_dy)) if (a_idx % 2 == 0) else (abs(raw_dx) > abs(raw_dy))
                        if prefer_x: step_dx = 1 if raw_dx > 0 else -1
                        else: step_dy = 1 if raw_dy > 0 else -1
                    elif raw_dx != 0: step_dx = 1 if raw_dx > 0 else -1
                    else: step_dy = 1 if raw_dy > 0 else -1
                
                # Basic collision avoidance
                proposed_pos = (ax + step_dx, ay + step_dy)
                if proposed_pos != obs["depot"]:
                    taken = set()
                    for prev_i, (pdx, pdy) in enumerate(actions):
                        pax, pay, _ps = obs["agent_states"][prev_i]
                        taken.add((pax + pdx, pay + pdy))
                    if proposed_pos in taken:
                        step_dx, step_dy = 0, 0 # Wait if blocked
                actions.append((step_dx, step_dy))

        if renderer:
            keep = renderer.render(obs)
            if not keep: done = True
            if save_frames_dir:
                try:
                    import pygame
                    os.makedirs(save_frames_dir, exist_ok=True)
                    pygame.image.save(renderer._screen, os.path.join(save_frames_dir, f"frame_{frame_idx:05d}.png"))
                except Exception: pass
            if clock and fps > 0: clock.tick(fps)
            frame_idx += 1

        obs, reward, done, _info = env.step(actions)
        total_reward += reward

    unserviced_capacity = 0
    if hasattr(env, "_state") and env._state is not None:
        unserviced_capacity = sum(d.c for d in env._state.demands)
    
    expired_capacity = 0.0
    if hasattr(env, "_episode_stats"):
        expired_capacity = env._episode_stats.get("expired_capacity", 0.0)

    return total_reward, total_initial_demand_capacity, unserviced_capacity + expired_capacity

def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    cfg = get_default_config()
    cfg.generator_params["total_demand"] = args.total_demand

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    static_tag = "_static" if args.static else ""
    run_name = f"{args.planner}{static_tag}_{run_timestamp}"
    
    checkpoint_dir = f"checkpoints/rl_generator/{run_name}"
    log_dir = f"runs/rl_generator/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir) if args.log_tb else None
    csv_log_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_log_path, 'w', newline='') as fh:
        csv.writer(fh).writerow(["episode", "serviced_reward", "gen_reward"])

    env = make_environment(cfg)
    planner = _init_planner(args.planner, cfg, device, args.planner_ckpt if args.planner == "model" else None)

    # Init Model
    model = DemandDiffusionModel(condition_dim=CONDITION_DIM, num_steps=args.max_steps)
    if args.init_diffusion_ckpt and os.path.exists(args.init_diffusion_ckpt):
        try:
            state = torch.load(args.init_diffusion_ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"[Init] Loaded checkpoint: {args.init_diffusion_ckpt}")
        except Exception as e:
            print(f"[Init] Failed to load checkpoint: {e}")
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    baseline = None
    renderer = None
    # [NEW] Early Stopping Variables
    best_gen_reward = -float('inf')
    patience_counter = 0

    if args.render or args.save_frames_dir:
        try:
            renderer = PygameRenderer(cfg.width, cfg.height, cell_size=24, caption="RL Training")
            renderer.init()
        except Exception: pass

    for ep in range(1, args.episodes + 1):
        # Depot Randomization
        if args.randomize_depot:
            rng = random.Random(args.seed + ep)
            new_depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))
            cfg.depot = new_depot
            env.depot = new_depot
            cfg.generator_params = {**cfg.generator_params, "depot": new_depot}

        # Prepare Condition
        cond_params = {f"param_{k}": v for k, v in cfg.generator_params.items()}
        condition = prepare_condition(cond_params).unsqueeze(0).to(device)

        # Generate Demands
        model.eval()
        with torch.no_grad():
            gen_tensor = model.sample(condition=condition, num_demands=int(cfg.generator_params["total_demand"]), grid_size=(cfg.width, cfg.height))
        
        # Decode using utils
        demands = decode_demands_from_tensor(gen_tensor, {
            'width': cfg.width, 'height': cfg.height, 'max_time': cfg.max_time,
            'max_c': cfg.generator_params['max_c'],
            'min_lifetime': cfg.generator_params['min_lifetime'],
            'max_lifetime': cfg.generator_params['max_lifetime']
        })

        if args.static:
            demands = apply_static_constraints(demands, cfg.max_time)

        # Rollout
        serviced_reward, total_demand_cap, total_failed_cap = _plan_episode(
            planner, env, demands, renderer=renderer, fps=args.fps, 
            save_frames_dir=args.save_frames_dir, debug=args.debug_planner
        )
        
        # Calculate Reward
        gen_reward = total_failed_cap - serviced_reward
        
        entropy_bonus = args.entropy_weight * calculate_spatial_entropy(demands)
        time_bonus = args.time_entropy_weight * calculate_temporal_entropy(demands)
        
        gen_reward += entropy_bonus + time_bonus

        # Early Stopping 
        # We track the raw generator reward to decide when to stop.
        if gen_reward > best_gen_reward:
            best_gen_reward = gen_reward
            patience_counter = 0
            # Save "best" model distinct from "latest"
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"[Early Stopping] No improvement for {args.patience} episodes. Best Reward: {best_gen_reward:.2f}")
            break

        # Log Heatmap
        if writer and ep % args.log_image_every == 0:
            log_density_heatmap(writer, ep, model, condition, cfg, device)

        if total_demand_cap < 10:
            gen_reward -= 100

        # Advantage
        if baseline is None: baseline = gen_reward
        adv = gen_reward - baseline
        baseline = args.baseline_beta * baseline + (1 - args.baseline_beta) * gen_reward

        if args.normalize_reward:
            adv_scaled = torch.tanh(torch.tensor(adv / (abs(baseline) + 1e-6), dtype=torch.float32, device=device))
        else:
            adv_scaled = torch.tensor(adv, dtype=torch.float32, device=device)

        # Update Model
        model.train()
        x_start = normalize_demands_for_training(demands, cfg).to(device).unsqueeze(0)
        
        noise, predicted_noise = model(x_start, condition)
        diff_loss = F.mse_loss(predicted_noise, noise)
        diff_loss_clipped = torch.clamp(diff_loss, max=args.diff_loss_clip)


        loss = diff_loss_clipped * (adv_scaled + args.sl_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Logging
        if writer:
            writer.add_scalar("reward/serviced", serviced_reward, ep)
            writer.add_scalar("reward/generator", gen_reward, ep)
            writer.add_scalar("reward/entropy_bonus", entropy_bonus, ep)
            writer.add_scalar("reward/time_bonus", time_bonus, ep)
            writer.add_scalar("train/advantage", adv, ep)
            writer.add_scalar("train/loss", loss.item(), ep)
            writer.add_scalar("train/diff_loss", diff_loss.item(), ep)
            writer.add_scalar("train/diff_loss_clipped", diff_loss_clipped.item(), ep)

        with open(csv_log_path, 'a', newline='') as fh:
            csv.writer(fh).writerow([ep, serviced_reward, gen_reward])

        print(f"[EP {ep:03d}] serviced={serviced_reward:.2f} gen={gen_reward:.2f} adv={adv:.2f} loss={loss.item():.4f}")

        if ep % args.save_every == 0 or ep == args.episodes:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"ckpt_ep_{ep}.pth"))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "latest.pth"))

    if renderer: renderer.close()
    if writer: writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
