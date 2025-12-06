"""
Generator Trainer Module

Adversarial training for diffusion generator to find planner weaknesses.
Uses RLGeneratorTrainer from training/generator/train_rl_diffusion_generator.py

Supports multi-GPU training with DDP:
- When num_gpus > 1, wraps model with DistributedDataParallel
- Each process trains on its own GPU
- Gradients are synchronized across GPUs
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import random
from copy import deepcopy
import torch
import torch.nn as nn

from .config import CoevolutionConfig
from .utils.distributed import (
    is_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    wrap_model_ddp,
    unwrap_model,
    reduce_tensor,
    barrier,
    print_rank0,
)

# Project imports
from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import CONDITION_DIM
from environment.env import GridEnvironment
from training.generator.train_rl_diffusion_generator import RLGeneratorTrainer
from training.generator.rl_utils import make_environment
from configs import get_default_config

# Default hyperparameters
DEFAULT_GEN_LR = 2e-6
DEFAULT_BASELINE_BETA = 0.9
DEFAULT_MAX_GRAD_NORM = 1.0


class GeneratorTrainer:
    """
    Adversarial trainer for diffusion generator.
    
    The goal is to train the generator to produce demand distributions
    that maximize difficulty for the planner (minimize planner reward).
    
    Wraps RLGeneratorTrainer from training/generator/train_rl_diffusion_generator.py
    with additional functionality for coevolution.
    
    Supports DDP for multi-GPU training when config.num_gpus > 1.
    """
    
    def __init__(
        self,
        config: CoevolutionConfig,
        planner_model: nn.Module,
        device: torch.device,
    ):
        self.config = config
        self.planner_model = planner_model  # StaticVRPModel for training
        self.device = device
        
        # Multi-GPU settings
        self.num_gpus = getattr(config, 'num_gpus', 1)
        self.local_rank = getattr(config, 'local_rank', 0)
        self.distributed = is_distributed()
        
        # Initialize diffusion model
        self._init_diffusion()
        
        # Wrap model with DDP if distributed
        if self.distributed:
            self.model = wrap_model_ddp(self.model, self.device)
            print_rank0(f"[GeneratorTrainer] Wrapped diffusion model with DDP on rank {self.local_rank}")
        
        # Initialize V2Planner for rollout (wraps the static model)
        self._init_v2_planner()
        
        # Build config object for RLGeneratorTrainer
        self._build_trainer_cfg()
        
        # Create environment
        self.env = GridEnvironment(
            width=config.env.map_size,
            height=config.env.map_size,
            num_agents=config.env.num_agents,
            capacity=config.env.capacity,
            depot=config.env.depot,
            max_time=config.env.max_time,
            max_end_time=config.env.max_end_time,
        )
        
        # Initialize RLGeneratorTrainer
        # Pass model into RLGeneratorTrainer. If distributed, pass the DDP-wrapped
        # module so that gradient synchronization happens automatically during
        # backward() in RLGeneratorTrainer. The trainer will create its own
        # optimizer from the passed model.parameters().
        model_for_trainer = self.model if self.distributed else self.model
        self.rl_trainer = RLGeneratorTrainer(
            model=model_for_trainer,
            planner=self.v2_planner,
            env=self.env,
            cfg=self.trainer_cfg,
            device=device,
            lr=DEFAULT_GEN_LR,
            baseline_beta=DEFAULT_BASELINE_BETA,
            normalize_reward=False,
            entropy_weight=0.80,
            time_entropy_weight=0.05,
            sl_weight=0.1,
            diff_loss_clip=10.0,
            static_mode=(config.mode == "static"),
            randomize_depot=config.env.randomize_depot,
        )
    
    def _build_trainer_cfg(self):
        """Build config object compatible with RLGeneratorTrainer."""
        cfg = self.config
        
        class TrainerCfg:
            pass
        
        trainer_cfg = get_default_config()
        
        self.trainer_cfg = TrainerCfg()
        self.trainer_cfg.width = cfg.env.map_size
        self.trainer_cfg.height = cfg.env.map_size
        self.trainer_cfg.max_time = cfg.env.max_time
        self.trainer_cfg.depot = cfg.env.depot
        self.trainer_cfg.generator_params = trainer_cfg.generator_params
        
    
    def _init_diffusion(self):
        """Initialize diffusion model."""
        self.model = DemandDiffusionModel(
            condition_dim=CONDITION_DIM,
            data_dim=5,
            time_emb_dim=64,
            num_steps=1000,
        ).to(self.device)
        
        # Load checkpoint if provided
        cfg = self.config
        if cfg.generator_checkpoint and os.path.exists(cfg.generator_checkpoint):
            state_dict = torch.load(cfg.generator_checkpoint, map_location=self.device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict, strict=False)
            print_rank0(f"[GeneratorTrainer] Loaded diffusion model from {cfg.generator_checkpoint}")
    
    def _init_v2_planner(self):
        """Initialize V2Planner for rollout evaluation."""
        from agent.planner.v2_planner import V2Planner
        
        cfg = self.config
        
        self.v2_planner = V2Planner(
            mode="static",
            device=str(self.device),
            grid_width=cfg.env.map_size,
            grid_height=cfg.env.map_size,
            full_capacity=cfg.env.capacity,
            max_time=cfg.env.max_time,
        )
        self._sync_planner_weights()
    
    def _sync_planner_weights(self):
        """Sync weights from planner_model to V2Planner."""
        self.v2_planner._ensure_model_loaded()
        self.v2_planner._model.load_state_dict(self.planner_model.state_dict(), strict=False)
        self.v2_planner._model.eval()
    
    def update_planner(self, planner_model: nn.Module):
        """Update the planner reference (after planner training)."""
        self.planner_model = planner_model
        self._sync_planner_weights()
        # Update RLGeneratorTrainer's planner reference
        self.rl_trainer.update_planner(self.v2_planner)
    
    def generate_demands(self, seed: Optional[int] = None) -> List[Tuple[int, int, int, int, int]]:
        """Generate demands using diffusion model."""
        return self.rl_trainer.generate_demands(seed=seed)
    
    def rollout_with_planner(
        self,
        demands: List[Tuple[int, int, int, int, int]],
        depot: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Rollout one episode with the planner and return total reward."""
        cfg = self.config
        
        if depot is None:
            depot = cfg.env.depot
        self.env.depot = depot
        
        serviced_reward, _, _ = self.rl_trainer.rollout(demands)
        return serviced_reward
    
    def train_step(self, seed: Optional[int] = None) -> Dict[str, float]:
        """One training step: generate -> rollout -> update."""
        cfg = self.config
        episode = seed if seed is not None else 0
        
        metrics = self.rl_trainer.train_step(
            episode=episode,
            seed=cfg.seed,
        )
        
        # Rename metrics for compatibility
        return {
            "diff_loss": metrics.get("diff_loss", 0.0),
            "advantage": metrics.get("advantage", 0.0),
            "gen_reward": metrics.get("gen_reward", 0.0),
            "planner_reward": metrics.get("serviced_reward", 0.0),
            "baseline": metrics.get("baseline", 0.0),
            "loss": metrics.get("loss", 0.0),
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        cfg = self.config
        n_episodes = cfg.episodes_per_epoch
        
        total_metrics = {}
        
        for ep in range(n_episodes):
            metrics = self.train_step(seed=cfg.seed + ep)
            
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            
            if (ep + 1) % 10 == 0:
                print_rank0(f"  Episode {ep+1}/{n_episodes}: "
                      f"planner_r={metrics['planner_reward']:.2f}, "
                      f"gen_r={metrics['gen_reward']:.2f}, "
                      f"loss={metrics['loss']:.4f}")
        
        avg_metrics = {k: v / n_episodes for k, v in total_metrics.items()}
        return avg_metrics
    
    def save_checkpoint(self, path: str, epoch: int, extra_state: Optional[Dict] = None):
        """Save generator checkpoint. Only saves on main process in distributed mode."""
        # Only save on main process
        if self.distributed and not is_main_process():
            barrier()
            return
            
        self.rl_trainer.save_checkpoint(path, epoch, extra_state)
        print_rank0(f"[GeneratorTrainer] Saved checkpoint to {path}")
        
        if self.distributed:
            barrier()
    
    def load_checkpoint(self, path: str):
        """Load generator checkpoint."""
        return self.rl_trainer.load_checkpoint(path)
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict for registry."""
        return self.rl_trainer.get_model_state_dict()
