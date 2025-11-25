from __future__ import annotations
from typing import Any
from configs import get_default_config
from environment.env import GridEnvironment
from agent.planner.rule_planner import RuleBasedPlanner
from agent.planner.model_planner import ModelPlanner
from agent.planner.cvrp_pomo_planner import CVRPPOMOPlanner
import copy
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
        wait_penalty_scale=float(getattr(cfg, "wait_penalty_scale", 0.001)),
        max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
        include_service_time=bool(getattr(cfg, "include_service_time", False)),
    )
    env.num_agents = cfg.num_agents
    return env, cfg

def build_planner(planner_type: str, cfg, device: torch.device, ckpt: str|None=None):
    full_cap = cfg.capacity
    if planner_type == 'greedy':
        return RuleBasedPlanner(full_capacity=full_cap)
    elif planner_type == 'model':
        planner_params = copy.deepcopy(getattr(cfg, 'model_planner_params', {}))
        planner_params.setdefault('device', str(device))
        ckpt = ckpt or planner_params.pop('ckpt', None)
        mp = ModelPlanner(full_capacity=full_cap, **planner_params)
        if ckpt and os.path.exists(ckpt):
            mp.load_from_ckpt(ckpt)
        return mp
    elif planner_type == 'cvrp_pomo':
        params = copy.deepcopy(cfg.cvrp_planner_params)
        params.pop('enabled', None)
        pomo_root = params.pop('pomo_root', None)
        if not pomo_root:
            raise ValueError('cvrp_pomo planner requires pomo_root in config.cvrp_planner_params')
        env_params = params.pop('env_params', {})
        model_params = params.pop('model_params', {})
        checkpoint = params.pop('checkpoint', None)
        device_override = params.pop('device', 'cpu')
        max_nodes = params.pop('max_nodes', env_params.get('problem_size', cfg.capacity))
        coord_norm = params.pop('coord_normalizer', None)
        selection_policy = params.pop('selection_policy', 'earliest_due')
        return CVRPPOMOPlanner(
            pomo_root=pomo_root,
            env_params=env_params,
            model_params=model_params,
            checkpoint=checkpoint,
            device=device_override,
            max_nodes=max_nodes,
            coord_normalizer=coord_norm,
            grid_width=cfg.width,
            grid_height=cfg.height,
            capacity=cfg.capacity,
            selection_policy=selection_policy,
            **params,
        )
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
