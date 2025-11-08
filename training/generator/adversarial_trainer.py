from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from environment.env import GridEnvironment
from agent.planner.base import AgentState
from agent.generator.base import Demand
from utils.pygame_renderer import PygameRenderer
from agent.generator.data_utils import unnormalize_value


@dataclass
class AdvConfig:
    randomize_depot: bool = True
    lr: float = 1e-4
    baseline_beta: float = 0.9
    normalize_reward: bool = False
    save_every: int = 10


def _generate_demands(model, condition, params: Dict[str, Any]):
    model.eval()
    with torch.no_grad():
        gen = model.sample(condition=condition, num_demands=int(params["total_demand"]), grid_size=(params["width"], params["height"]))
    width = params["width"]; height = params["height"]; max_time = params["max_time"]; max_c = params["max_c"]
    min_lifetime = params["min_lifetime"]; max_lifetime = params["max_lifetime"]
    demands: List[Tuple[int,int,int,int,int]] = []
    for row in gen.cpu().numpy():
        t_raw, x_raw, y_raw, c_raw, life_raw = row
        t_val = int(round(unnormalize_value(t_raw, 0, max_time - 1)))
        x_val = int(round(unnormalize_value(x_raw, 0, width - 1)))
        y_val = int(round(unnormalize_value(y_raw, 0, height - 1)))
        c_val = int(round(unnormalize_value(c_raw, 1, max_c)))
        life_val = int(round(unnormalize_value(life_raw, min_lifetime, max_lifetime)))
        t_val = max(0, min(max_time - 1, t_val))
        x_val = max(0, min(width - 1, x_val))
        y_val = max(0, min(height - 1, y_val))
        c_val = max(1, min(max_c, c_val))
        life_val = max(min_lifetime, min(max_lifetime, life_val))
        end_t = t_val + life_val
        demands.append((x_val, y_val, t_val, c_val, end_t))
    return demands


def rollout_episode(planner, env: GridEnvironment, demands: List[Tuple[int,int,int,int,int]], *, renderer: PygameRenderer|None=None, fps: int=10) -> float:
    obs = env.reset()
    if hasattr(env, "_state") and env._state is not None:
        env._state.demands.extend([Demand(x=d[0], y=d[1], t=d[2], c=d[3], end_t=d[4]) for d in demands])
    total_reward = 0.0
    done = False
    clock = None
    if renderer is not None:
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
        for a_idx, queue in enumerate(plans):
            if len(queue) == 0:
                actions.append((0,0)); continue
            tx, ty = queue[0]
            ax, ay, _s = obs["agent_states"][a_idx]
            if tx != ax:
                actions.append((1 if tx>ax else -1, 0))
            elif ty != ay:
                actions.append((0, 1 if ty>ay else -1))
            else:
                actions.append((0,0))
        if renderer is not None:
            keep = renderer.render(obs)
            if not keep:
                done = True
            if clock is not None and fps > 0:
                clock.tick(fps)
        obs, reward, done, _ = env.step(actions)
        total_reward += reward
    return total_reward


class DiffusionAdversarialTrainer:
    def __init__(self, env: GridEnvironment, model, condition, cfg, device: torch.device, adv_cfg: AdvConfig) -> None:
        self.env = env
        self.model = model
        self.condition = condition
        self.cfg = cfg
        self.device = device
        self.adv_cfg = adv_cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=adv_cfg.lr)
        self.baseline = None

    def train(self, planner, episodes: int, renderer: PygameRenderer|None=None, save_path: str|None=None, seed: int = 1) -> None:
        import random
        for ep in range(1, episodes+1):
            if self.adv_cfg.randomize_depot:
                rng = random.Random(seed + ep)
                new_depot = (rng.randint(0, self.cfg.width-1), rng.randint(0, self.cfg.height-1))
                self.cfg.depot = new_depot
                self.env.depot = new_depot
                self.cfg.generator_params = {**self.cfg.generator_params, "depot": new_depot}
            demands = _generate_demands(self.model, self.condition, {
                'width': self.cfg.width,
                'height': self.cfg.height,
                'max_time': self.cfg.max_time,
                'max_c': self.cfg.generator_params['max_c'],
                'min_lifetime': self.cfg.generator_params['min_lifetime'],
                'max_lifetime': self.cfg.generator_params['max_lifetime'],
                'total_demand': self.cfg.generator_params['total_demand']
            })
            env_reward = rollout_episode(planner, self.env, demands, renderer=renderer)
            gen_reward = -env_reward
            if self.baseline is None:
                self.baseline = gen_reward
            adv = gen_reward - self.baseline
            self.baseline = self.adv_cfg.baseline_beta * self.baseline + (1 - self.adv_cfg.baseline_beta) * gen_reward
            if self.adv_cfg.normalize_reward:
                adv_scaled = torch.tanh(torch.tensor(adv / (abs(self.baseline) + 1e-6), dtype=torch.float32, device=self.device))
            else:
                adv_scaled = torch.tensor(adv, dtype=torch.float32, device=self.device)
            # build x_start
            dem_tensor = []
            max_time = self.cfg.max_time - 1
            width = self.cfg.width - 1
            height = self.cfg.height - 1
            max_c = self.cfg.generator_params['max_c']
            min_life = self.cfg.generator_params['min_lifetime']
            max_life = self.cfg.generator_params['max_lifetime']
            for (x,y,t,c,end_t) in demands:
                lifetime = end_t - t
                norm_t = (t - 0) / max_time
                norm_x = (x - 0) / width
                norm_y = (y - 0) / height
                norm_c = (c - 1) / (max_c - 1 if max_c > 1 else 1)
                norm_life = (lifetime - min_life) / (max_life - min_life if max_life > min_life else 1)
                dem_tensor.append([norm_t, norm_x, norm_y, norm_c, norm_life])
            if not dem_tensor:
                dem_tensor.append([0,0,0,0,0])
            x_start = torch.tensor(dem_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            noise, predicted_noise = self.model(x_start, self.condition)
            diff_loss = F.mse_loss(predicted_noise, noise)
            loss = diff_loss * adv_scaled
            self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.opt.step()
            print(f"[EP {ep:03d}] env={env_reward:.2f} gen={gen_reward:.2f} adv={adv:.2f} diff={diff_loss.item():.4f} loss={loss.item():.4f}")
            if save_path and (ep % self.adv_cfg.save_every == 0 or ep == episodes):
                import os
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
