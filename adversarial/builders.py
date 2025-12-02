from __future__ import annotations
from typing import Any
from configs import get_default_config
from environment.env import GridEnvironment
from agent.planner.rule_planner import RuleBasedPlanner
from agent.planner.v2_planner import V2Planner, create_v2_planner
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
        depot_return_bonus_scale=float(getattr(cfg, "depot_return_bonus_scale", 0.0)),
        max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
        include_service_time=bool(getattr(cfg, "include_service_time", False)),
    )
    env.num_agents = cfg.num_agents
    return env, cfg

def build_planner(planner_type: str, cfg, device: torch.device, ckpt: str|None=None):
    full_cap = cfg.capacity
    if planner_type == 'greedy':
        return RuleBasedPlanner(full_capacity=full_cap)
    elif planner_type in ('model', 'dynamic', 'static'):
        # Use V2Planner (POMO-based architecture)
        v2_params = copy.deepcopy(getattr(cfg, 'v2_planner_params', {}))
        mode = 'static' if planner_type == 'static' else 'dynamic'
        static_ckpt = ckpt or v2_params.pop('static_ckpt', 'checkpoints/static_vrp_v2/best_n20.pt')
        adapter_ckpt = v2_params.pop('adapter_ckpt', 'checkpoints/dynamic_adapter_v2/best_adapter.pt')
        # Remove device from v2_params if present to avoid duplicate argument
        v2_params.pop('device', None)
        return create_v2_planner(
            mode=mode,
            static_checkpoint=static_ckpt,
            adapter_checkpoint=adapter_ckpt if mode == 'dynamic' else None,
            device=str(device),
            grid_width=cfg.width,
            grid_height=cfg.height,
            full_capacity=full_cap,
            max_time=cfg.max_time,
            **v2_params,
        )
    else:
        raise ValueError(f'Unsupported planner_type {planner_type}. Use "greedy", "static", "dynamic", or "model".')


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


def get_planner_trainable_params(planner):
    """
    Get trainable parameters from a V2Planner.
    
    For V2Planner in dynamic mode: returns adapter parameters
    For V2Planner in static mode: returns all model parameters
    
    Returns:
        list of parameters or None if not trainable
    """
    if isinstance(planner, V2Planner):
        planner._ensure_model_loaded()
        if planner.mode == "dynamic":
            # Get adapter parameters from DynamicVRPModel
            return planner._model.get_trainable_params()
        else:
            # Static mode - can finetune whole model
            return list(planner._model.parameters())
    return None


def build_planner_optimizer(planner, lr: float = 1e-4, weight_decay: float = 1e-6):
    """
    Build optimizer for planner training.
    
    Args:
        planner: V2Planner instance
        lr: learning rate
        weight_decay: weight decay
        
    Returns:
        optimizer or None if planner is not trainable
    """
    params = get_planner_trainable_params(planner)
    if params:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return None


def save_planner_checkpoint(planner, path: str, epoch: int = 0, extra_state: dict | None = None):
    """
    Save planner checkpoint.
    
    For V2Planner: saves static model or adapter state depending on mode.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    state = {'epoch': epoch}
    if extra_state:
        state.update(extra_state)
    
    if isinstance(planner, V2Planner):
        planner._ensure_model_loaded()
        if planner.mode == "dynamic":
            state['adapter_state'] = planner._model.adapter_state_dict()
            state['mode'] = 'dynamic'
        else:
            state['model_state_dict'] = planner._model.state_dict()
            state['mode'] = 'static'
    
    torch.save(state, path)
    print(f'[Planner] Saved checkpoint to {path}')
