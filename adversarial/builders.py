from __future__ import annotations
from typing import Any
from configs import get_default_config
from environment.env import GridEnvironment
from agent.planner.rule_planner import RuleBasedPlanner
from agent.planner.model_planner import ModelPlanner
from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import prepare_condition, CONDITION_DIM
import torch, os


def build_env(cfg=None):
    cfg = cfg or get_default_config()
    env = GridEnvironment(
        width=cfg.width,
        height=cfg.height,
        num_agents=cfg.num_agents,
        capacity=cfg.capacity,
        depot=cfg.depot,
        max_time=cfg.max_time,
        expiry_penalty_scale=float(getattr(cfg, "expiry_penalty_scale", 5.0)),
        switch_penalty_scale=float(getattr(cfg, "switch_penalty_scale", 0.01)),
        capacity_reward_scale=float(getattr(cfg, "capacity_reward_scale", 10.0)),
        exploration_history_n=int(getattr(cfg, "exploration_history_n", 0)),
        exploration_penalty_scale=float(getattr(cfg, "exploration_penalty_scale", 0.0)),
    )
    env.num_agents = cfg.num_agents
    return env, cfg

def build_planner(planner_type: str, cfg, device: torch.device, ckpt: str|None=None):
    full_cap = cfg.capacity
    if planner_type == 'greedy':
        return RuleBasedPlanner(full_capacity=full_cap)
    elif planner_type == 'model':
        mp = ModelPlanner(d_model=cfg.model_planner_params.get('d_model',128),
                          nhead=cfg.model_planner_params.get('nhead',8),
                          nlayers=cfg.model_planner_params.get('nlayers',2),
                          time_plan=cfg.model_planner_params.get('time_plan',3),
                          lateness_lambda=cfg.model_planner_params.get('lateness_lambda',0.0),
                          device=str(device),
                          full_capacity=full_cap)
        ckpt = ckpt or cfg.model_planner_params.get('ckpt')
        if ckpt and os.path.exists(ckpt):
            mp.load_from_ckpt(ckpt)
        return mp
    else:
        raise ValueError(f'Unsupported planner_type {planner_type}')


def build_diffusion(cfg, device: torch.device, init_ckpt: str|None=None, num_steps: int = 1000):
    cond_params = {f'param_{k}': v for k,v in cfg.generator_params.items()}
    condition = prepare_condition(cond_params).unsqueeze(0).to(device)
    model = DemandDiffusionModel(condition_dim=CONDITION_DIM, num_steps=num_steps)
    if init_ckpt and os.path.exists(init_ckpt):
        try:
            state = torch.load(init_ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f'[Diffusion] load failed: {e}')
    model.to(device)
    return model, condition
