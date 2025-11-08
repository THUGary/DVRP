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
python scripts/train_rl_diffusion_generator.py --episodes 50 --planner greedy --device cuda
python scripts/train_rl_diffusion_generator.py --episodes 50 --planner model --planner_ckpt checkpoints/planner/planner_20_2_200.pt --device cuda

Outputs:
- Updated diffusion model checkpoint at --out_ckpt.
- CSV log with episode, env_reward, gen_reward.
"""
from __future__ import annotations
import argparse
import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F

import sys
import pathlib
# Robust project root discovery (search upward for configs.py)
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.generator_model.diffusion_model import DemandDiffusionModel  # type: ignore
from agent.generator.data_utils import prepare_condition, unnormalize_value, CONDITION_DIM  # type: ignore
from environment.env import GridEnvironment  # type: ignore
from agent.planner.rule_planner import RuleBasedPlanner  # type: ignore
from agent.planner.model_planner import ModelPlanner  # type: ignore
from agent.planner.base import AgentState  # type: ignore
from utils.pygame_renderer import PygameRenderer  # type: ignore
from configs import get_default_config  # type: ignore
import random


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adversarial RL training for diffusion demand generator against a planner")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--planner", type=str, default="greedy", choices=["greedy", "model"], help="Planner to adversarially attack")
    p.add_argument("--planner_ckpt", type=str, default="checkpoints/planner/planner_20_2_200.pt", help="Checkpoint for model planner if --planner model")
    p.add_argument("--total_demand", type=int, default=50, help="Number of demands to generate per episode")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_ckpt", type=str, default="checkpoints/diffusion_model_adv.pth")
    p.add_argument("--init_diffusion_ckpt", type=str, default="checkpoints/diffusion_model.pth", help="Initialize diffusion model from this checkpoint if exists")
    p.add_argument("--log_csv", type=str, default="runs/diffusion_adv_rewards.csv")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=1000, help="Diffusion model internal steps (num_steps) if reinit")
    p.add_argument("--normalize_reward", action="store_true", help="Normalize generator rewards across batch for stable updates")
    p.add_argument("--baseline_beta", type=float, default=0.9, help="EMA baseline beta for variance reduction")
    p.add_argument("--render", action="store_true", help="Render environment with Pygame during rollout")
    p.add_argument("--fps", type=int, default=10, help="Render FPS; used only with --render")
    p.add_argument("--save_frames_dir", type=str, default="", help="If set, save rendered frames as PNGs to this directory (headless)")
    p.add_argument("--randomize_depot", action="store_true", help="Randomize depot per episode (match run.py behavior)")
    p.add_argument("--debug_planner", action="store_true", help="Print per-step planner debug info (active demands, chosen targets, actions)")
    return p


def _make_environment(cfg) -> GridEnvironment:
    env = GridEnvironment(width=cfg.width, height=cfg.height, num_agents=cfg.num_agents, capacity=cfg.capacity, depot=cfg.depot, max_time=cfg.max_time)
    env.num_agents = cfg.num_agents
    return env


def _init_planner(planner_type: str, cfg, device: torch.device, ckpt_path: str | None) -> Any:
    full_cap = cfg.capacity
    if planner_type == "greedy":
        return RuleBasedPlanner(full_capacity=full_cap)
    elif planner_type == "model":
        mp = ModelPlanner(d_model=cfg.model_planner_params.get("d_model",128),
                          nhead=cfg.model_planner_params.get("nhead",8),
                          nlayers=cfg.model_planner_params.get("nlayers",2),
                          time_plan=cfg.model_planner_params.get("time_plan",3),
                          lateness_lambda=cfg.model_planner_params.get("lateness_lambda",0.0),
                          device=str(device),
                          full_capacity=full_cap)
        ckpt = ckpt_path or cfg.model_planner_params.get("ckpt")
        if ckpt and os.path.exists(ckpt):
            mp.load_from_ckpt(ckpt)
            print(f"[Planner] Loaded model planner checkpoint: {ckpt}")
        else:
            print(f"[Planner] WARNING: checkpoint not found at {ckpt}; using random weights.")
        return mp
    else:
        raise ValueError(f"Unsupported planner type: {planner_type}")


def _plan_episode(planner, env: GridEnvironment, demands: List[Tuple[int,int,int,int,int]], *, renderer: PygameRenderer | None = None, fps: int = 10, save_frames_dir: str = "", debug: bool = False) -> float:
    """Inject demands then roll out a simple control loop with no sophisticated planner replanning every step.
    For adversarial training we only need the resulting reward.
    We simulate naive control: agents move greedily towards first planned target each step.
    """
    obs = env.reset()
    # Pre-insert demands into generator-less environment state
    # We'll directly extend env._state.demands if accessible (adversarial setting)
    if hasattr(env, "_state") and env._state is not None:
        # convert to Demand objects via a minimal import
        from agent.generator.base import Demand
        env._state.demands.extend([Demand(x=d[0], y=d[1], t=d[2], c=d[3], end_t=d[4]) for d in demands])
    total_reward = 0.0
    done = False
    frame_idx = 0
    clock = None
    if renderer is not None:
        try:
            import pygame
            clock = pygame.time.Clock()
        except Exception:
            clock = None
    step_count = 0
    while not done:
        # observations demands format: (x,y,t,c,end_t)
        obs_demands = obs["demands"]
        agent_states = [AgentState(x=a[0], y=a[1], s=a[2]) for a in obs["agent_states"]]
        depot = tuple(obs["depot"])
        # plan short horizon targets
        plans = planner.plan(observations=obs_demands, agent_states=agent_states, depot=depot, t=obs["time"], horizon=1)
        actions = []
        for a_idx, queue in enumerate(plans):
            if len(queue) == 0:
                actions.append((0,0))
            else:
                tx, ty = queue[0]
                ax, ay, _s = obs["agent_states"][a_idx]
                raw_dx = tx - ax
                raw_dy = ty - ay
                step_dx, step_dy = 0, 0
                # If both axes need movement choose one axis to reduce diagonal crowding.
                if raw_dx != 0 or raw_dy != 0:
                    if raw_dx != 0 and raw_dy != 0:
                        # Axis selection strategy: alternate agents to diversify paths.
                        prefer_x = (abs(raw_dx) >= abs(raw_dy)) if (a_idx % 2 == 0) else (abs(raw_dx) > abs(raw_dy))
                        if prefer_x:
                            step_dx = 1 if raw_dx > 0 else -1
                        else:
                            step_dy = 1 if raw_dy > 0 else -1
                    elif raw_dx != 0:
                        step_dx = 1 if raw_dx > 0 else -1
                    else:
                        step_dy = 1 if raw_dy > 0 else -1
                # Collision avoidance: avoid proposing a cell already selected by a previous agent (excluding depot).
                proposed_pos = (ax + step_dx, ay + step_dy)
                if proposed_pos != obs["depot"]:
                    taken = set()
                    for prev_i, (pdx, pdy) in enumerate(actions):
                        pax, pay, _ps = obs["agent_states"][prev_i]
                        taken.add((pax + pdx, pay + pdy))
                    if proposed_pos in taken:
                        # Try alternate axis if possible.
                        if step_dx != 0 and raw_dy != 0:
                            alt_pos = (ax, ay + (1 if raw_dy > 0 else -1))
                            if alt_pos not in taken:
                                step_dx, step_dy = 0, 1 if raw_dy > 0 else -1
                            else:
                                step_dx, step_dy = 0, 0  # stay
                        elif step_dy != 0 and raw_dx != 0:
                            alt_pos = (ax + (1 if raw_dx > 0 else -1), ay)
                            if alt_pos not in taken:
                                step_dx, step_dy = 1 if raw_dx > 0 else -1, 0
                            else:
                                step_dx, step_dy = 0, 0
                        else:
                            step_dx, step_dy = 0, 0
                actions.append((step_dx, step_dy))
        if debug and step_count < 15:  # limit debug spam
            print(f"[DEBUG] t={obs['time']} active_demands={len(obs_demands)}")
            if obs_demands:
                sample_demands = obs_demands[:5]
                print(f"[DEBUG] sample_demands (x,y,t,c,due)={sample_demands}")
            plan_targets = [list(q)[:1] for q in plans]
            print(f"[DEBUG] first_targets={plan_targets}")
            print(f"[DEBUG] actions={actions}")
        # optional render
        if renderer is not None:
            keep = renderer.render(obs)
            if not keep:
                # window closed; stop early
                done = True
            # optionally save frame
            if save_frames_dir:
                try:
                    import pygame
                    import os as _os
                    _os.makedirs(save_frames_dir, exist_ok=True)
                    pygame.image.save(renderer._screen, _os.path.join(save_frames_dir, f"frame_{frame_idx:05d}.png"))
                except Exception:
                    pass
            if clock is not None and fps > 0:
                clock.tick(fps)
            frame_idx += 1

        obs, reward, done, _info = env.step(actions)
        total_reward += reward
        step_count += 1
    return total_reward


def _generate_demands(model: DemandDiffusionModel, condition: torch.Tensor, params: Dict[str, Any], device: torch.device) -> List[Tuple[int,int,int,int,int]]:
    model.eval()
    with torch.no_grad():
        gen = model.sample(condition=condition, num_demands=int(params["total_demand"]), grid_size=(params["width"], params["height"]))  # [num_dem,5]
    # Unnormalize each demand following ranges
    width = params["width"]
    height = params["height"]
    max_time = params["max_time"]
    max_c = params["max_c"]
    min_lifetime = params["min_lifetime"]
    max_lifetime = params["max_lifetime"]
    demands: List[Tuple[int,int,int,int,int]] = []
    for row in gen.cpu().numpy():
        t_raw, x_raw, y_raw, c_raw, life_raw = row
        t_val = int(round(unnormalize_value(t_raw, 0, max_time - 1)))
        x_val = int(round(unnormalize_value(x_raw, 0, width - 1)))
        y_val = int(round(unnormalize_value(y_raw, 0, height - 1)))
        c_val = int(round(unnormalize_value(c_raw, 1, max_c)))
        life_val = int(round(unnormalize_value(life_raw, min_lifetime, max_lifetime)))
        # clip constraints
        t_val = max(0, min(max_time - 1, t_val))
        x_val = max(0, min(width - 1, x_val))
        y_val = max(0, min(height - 1, y_val))
        c_val = max(1, min(max_c, c_val))
        life_val = max(min_lifetime, min(max_lifetime, life_val))
        end_t = t_val + life_val
        demands.append((x_val, y_val, t_val, c_val, end_t))
    return demands


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    cfg = get_default_config()
    # override total demand if provided
    cfg.generator_params["total_demand"] = args.total_demand

    env = _make_environment(cfg)
    planner = _init_planner(args.planner, cfg, device, args.planner_ckpt if args.planner == "model" else None)

    # prepare condition vector from generator params
    cond_params = {f"param_{k}": v for k, v in cfg.generator_params.items()}
    condition = prepare_condition(cond_params).unsqueeze(0).to(device)

    # initialize diffusion model (could load existing adv weights to continue training)
    model = DemandDiffusionModel(condition_dim=CONDITION_DIM, num_steps=args.max_steps)
    # Try to load initialization weights
    init_ckpt = args.init_diffusion_ckpt
    if init_ckpt and os.path.exists(init_ckpt):
        try:
            state = torch.load(init_ckpt, map_location=device)
            # handle DataParallel checkpoints (state_dict directly)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[Init] Loaded diffusion checkpoint: {init_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            print(f"[Init] WARNING: failed to load diffusion checkpoint {init_ckpt}: {e}")
    else:
        print(f"[Init] No diffusion init checkpoint found at {init_ckpt}; starting from random init.")
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    baseline = None
    os.makedirs(os.path.dirname(args.out_ckpt) or '.', exist_ok=True)
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or '.', exist_ok=True)
        if not os.path.exists(args.log_csv):
            with open(args.log_csv, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['episode','env_reward','gen_reward'])

    renderer = None
    # Try to enable headless rendering if rendering/saving frames without a display
    try:
        import os as _os
        if (args.render or args.save_frames_dir) and not _os.environ.get("DISPLAY"):
            _os.environ["SDL_VIDEODRIVER"] = "dummy"
    except Exception:
        pass
    if args.render or args.save_frames_dir:
        try:
            renderer = PygameRenderer(cfg.width, cfg.height, cell_size=24, caption="RL Diffusion Training")
            renderer.init()
        except Exception as e:
            print(f"[Render] Failed to init renderer: {e}")
            renderer = None

    for ep in range(1, args.episodes + 1):
        # Optional depot randomization per episode (like run.py)
        if args.randomize_depot:
            rng = random.Random(args.seed + ep)
            new_depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))
            cfg.depot = new_depot
            env.depot = new_depot
            cfg.generator_params = {**cfg.generator_params, "depot": new_depot}
        # Sample demands (forward diffusion training step): we need synthetic training target x_start
        # For adversarial update we can treat model.sample output as x_start approximation.
        demands = _generate_demands(model, condition, {
            'width': cfg.width,
            'height': cfg.height,
            'max_time': cfg.max_time,
            'max_c': cfg.generator_params['max_c'],
            'min_lifetime': cfg.generator_params['min_lifetime'],
            'max_lifetime': cfg.generator_params['max_lifetime'],
            'total_demand': cfg.generator_params['total_demand']
        }, device)

        # Evaluate planner reward under these demands
        env_reward = _plan_episode(planner, env, demands, renderer=renderer, fps=args.fps, save_frames_dir=args.save_frames_dir, debug=args.debug_planner)
        gen_reward = -env_reward  # adversarial objective

        if baseline is None:
            baseline = gen_reward
        adv = gen_reward - baseline
        baseline = args.baseline_beta * baseline + (1 - args.baseline_beta) * gen_reward

        if args.normalize_reward:
            adv_scaled = torch.tanh(torch.tensor(adv / (abs(baseline) + 1e-6), dtype=torch.float32, device=device))
        else:
            adv_scaled = torch.tensor(adv, dtype=torch.float32, device=device)

        # Construct a pseudo training batch: we treat current demands tensor as x_start
        # Re-normalize demands into model space to compute noise-prediction loss
        # Build normalized tensor (B=1, N, 5)
        dem_tensor = []
        max_time = cfg.max_time - 1
        width = cfg.width - 1
        height = cfg.height - 1
        max_c = cfg.generator_params['max_c']
        min_life = cfg.generator_params['min_lifetime']
        max_life = cfg.generator_params['max_lifetime']
        for (x,y,t,c,end_t) in demands:
            lifetime = end_t - t
            norm_t = (t - 0) / max_time
            norm_x = (x - 0) / width
            norm_y = (y - 0) / height
            norm_c = (c - 1) / (max_c - 1 if max_c > 1 else 1)
            norm_life = (lifetime - min_life) / (max_life - min_life if max_life > min_life else 1)
            dem_tensor.append([norm_t, norm_x, norm_y, norm_c, norm_life])
        if not dem_tensor:
            # if no demands (unlikely), create a dummy row
            dem_tensor.append([0,0,0,0,0])
        x_start = torch.tensor(dem_tensor, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,5]

        noise, predicted_noise = model(x_start, condition)
        diff_loss = F.mse_loss(predicted_noise, noise)
        loss = diff_loss * adv_scaled

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if args.log_csv:
            with open(args.log_csv, 'a', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([ep, env_reward, gen_reward])

        print(f"[EP {ep:03d}] env_reward={env_reward:.2f} gen_reward={gen_reward:.2f} adv={adv:.2f} loss={loss.item():.4f} diff={diff_loss.item():.4f}")

        if ep % args.save_every == 0 or ep == args.episodes:
            torch.save(model.state_dict(), args.out_ckpt)
            print(f"[CKPT] saved diffusion adversarial weights -> {args.out_ckpt}")

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    print("Training complete.")


if __name__ == "__main__":
    main()
