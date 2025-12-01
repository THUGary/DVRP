"""
Generator Trainer Module

Adversarial training for diffusion generator to find planner weaknesses.
Reuses existing functions from training/generator/adversarial_trainer.py
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoevolutionConfig
from .utils.demand_converter import DemandConverter, DemandTuple

# Project imports
from configs import COORD_NORM, DEMAND_NORM
from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import prepare_condition, CONDITION_DIM
from environment.env import GridEnvironment
from training.generator.adversarial_trainer import (
    _generate_demands,
    rollout_episode,
    AdvConfig,
    DiffusionAdversarialTrainer,
)

# Default hyperparameters from AdvConfig
DEFAULT_GEN_LR = AdvConfig.lr  # 1e-4
DEFAULT_BASELINE_BETA = AdvConfig.baseline_beta  # 0.9
DEFAULT_MAX_GRAD_NORM = 1.0


class GeneratorTrainer:
    """
    Adversarial trainer for diffusion generator.
    
    The goal is to train the generator to produce demand distributions
    that maximize difficulty for the planner (minimize planner reward).
    
    Reuses the DiffusionAdversarialTrainer from training/generator/adversarial_trainer.py
    but with additional wrapper functionality for coevolution.
    
    NOTE: For rollout evaluation, we use V2Planner which wraps the StaticVRPModel
    and provides the plan() interface expected by rollout_episode().
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
        
        # Initialize diffusion model
        self._init_diffusion()
        self._init_optimizer()
        
        # Initialize V2Planner for rollout (wraps the static model)
        self._init_v2_planner()
        
        # Environment for rollout (reused by adversarial_trainer)
        self.env = GridEnvironment(
            width=config.env.map_size,
            height=config.env.map_size,
            num_agents=config.env.num_agents,
            capacity=config.env.capacity,
            depot=config.env.depot,
            max_time=config.env.max_time,
            max_end_time=config.env.max_end_time,
        )
        
        # Build a config-like object for adversarial_trainer compatibility
        self._build_trainer_cfg()
        
        # Baseline for variance reduction (from AdvConfig)
        self.baseline = None
        self.baseline_beta = DEFAULT_BASELINE_BETA
        
        # Condition tensor
        self.condition = prepare_condition({}).unsqueeze(0).to(device)
    
    def _build_trainer_cfg(self):
        """Build config object compatible with DiffusionAdversarialTrainer."""
        cfg = self.config
        
        # Create a simple namespace-like object
        class TrainerCfg:
            pass
        
        self.trainer_cfg = TrainerCfg()
        self.trainer_cfg.map_size = cfg.env.map_size
        self.trainer_cfg.max_time = cfg.env.max_time
        self.trainer_cfg.depot = cfg.env.depot
        self.trainer_cfg.generator_params = {
            'max_c': cfg.env.max_c,
            'min_lifetime': cfg.env.min_lifetime,
            'max_lifetime': cfg.env.max_lifetime,
            'total_demand': cfg.env.total_demand,
            'depot': cfg.env.depot,
        }
    
    def _init_diffusion(self):
        """Initialize diffusion model using defaults from DemandDiffusionModel."""
        # Default model architecture from models/generator_model/diffusion_model.py
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
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[GeneratorTrainer] Loaded diffusion model from {cfg.generator_checkpoint}")
    
    def _init_optimizer(self):
        """Initialize optimizer using default hyperparameters from AdvConfig."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=DEFAULT_GEN_LR,
        )
    
    def _init_v2_planner(self):
        """Initialize V2Planner for rollout evaluation."""
        from agent.planner.v2_planner import V2Planner
        
        cfg = self.config
        
        # Create V2Planner in static mode (wraps the static model)
        self.v2_planner = V2Planner(
            mode="static",
            device=str(self.device),
            grid_width=cfg.env.map_size,
            grid_height=cfg.env.map_size,
            full_capacity=cfg.env.capacity,
            max_time=cfg.env.max_time,
        )
        # Load the static model weights into V2Planner
        self._sync_planner_weights()
    
    def _sync_planner_weights(self):
        """Sync weights from planner_model to V2Planner."""
        # Ensure V2Planner's model is loaded
        self.v2_planner._ensure_model_loaded()
        # Copy state dict from the training model
        self.v2_planner._model.load_state_dict(self.planner_model.state_dict())
        self.v2_planner._model.eval()
    
    def update_planner(self, planner_model: nn.Module):
        """Update the planner reference (after planner training)."""
        self.planner_model = planner_model
        # Sync weights to V2Planner for rollout
        self._sync_planner_weights()
    
    def generate_demands(self, seed: Optional[int] = None) -> List[Tuple[int, int, int, int, int]]:
        """
        Generate demands using diffusion model.
        Reuses _generate_demands from adversarial_trainer.py
        """
        cfg = self.config
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # For static mode, we need to modify the output to have t=0
        demands = _generate_demands(self.model, self.condition, {
            'map_size': cfg.env.map_size,
            'max_time': cfg.env.max_time,
            'max_c': cfg.env.max_c,
            'min_lifetime': cfg.env.min_lifetime,
            'max_lifetime': cfg.env.max_lifetime,
            'total_demand': cfg.env.total_demand,
        })
        
        # Convert to static demands if in static mode
        if cfg.mode == "static":
            # All demands at t=0 with deadline=max_time
            demands = [(x, y, 0, c, cfg.env.max_time) for (x, y, t, c, end_t) in demands]
        
        return demands
    
    def rollout_with_planner(
        self,
        demands: List[Tuple[int, int, int, int, int]],
        depot: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Rollout one episode with the planner and return total reward.
        Reuses rollout_episode from adversarial_trainer.py
        """
        cfg = self.config
        
        # Set depot
        if depot is None:
            depot = cfg.env.depot
        self.env.depot = depot
        
        # Use existing rollout_episode function with V2Planner (has plan() method)
        return rollout_episode(self.v2_planner, self.env, demands)
    
    def compute_adversarial_loss(
        self,
        demands: List[Tuple[int, int, int, int, int]],
        planner_reward: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute adversarial loss for generator update.
        Reuses the loss computation logic from DiffusionAdversarialTrainer.train()
        """
        cfg = self.config
        
        # Generator reward (negative of planner reward)
        gen_reward = -planner_reward
        
        # Update baseline with EMA
        if self.baseline is None:
            self.baseline = gen_reward
        adv = gen_reward - self.baseline
        self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * gen_reward
        
        adv_scaled = torch.tensor(adv, dtype=torch.float32, device=self.device)
        
        # Build x_start tensor from demands (same logic as adversarial_trainer.py)
        dem_tensor = []
        max_time = cfg.env.max_time - 1
        map_max = cfg.env.map_size - 1  # Square map
        max_c = cfg.env.max_c
        min_life = cfg.env.min_lifetime
        max_life = cfg.env.max_lifetime
        
        for (x, y, t, c, end_t) in demands:
            lifetime = end_t - t
            norm_t = (t - 0) / max_time if max_time > 0 else 0
            norm_x = (x - 0) / map_max if map_max > 0 else 0
            norm_y = (y - 0) / map_max if map_max > 0 else 0
            norm_c = (c - 1) / (max_c - 1) if max_c > 1 else 0
            norm_life = (lifetime - min_life) / (max_life - min_life) if max_life > min_life else 0
            dem_tensor.append([norm_t, norm_x, norm_y, norm_c, norm_life])
        
        if not dem_tensor:
            dem_tensor.append([0, 0, 0, 0, 0])
        
        x_start = torch.tensor(dem_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Forward through diffusion model to get loss
        self.model.train()
        noise, predicted_noise = self.model(x_start, self.condition)
        diff_loss = F.mse_loss(predicted_noise, noise)
        
        # Scale by advantage
        loss = diff_loss * adv_scaled
        
        return loss, {
            "diff_loss": diff_loss.item(),
            "advantage": adv,
            "gen_reward": gen_reward,
            "planner_reward": planner_reward,
            "baseline": self.baseline,
        }
    
    def train_step(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        One training step: generate -> rollout -> update.
        """
        cfg = self.config
        
        # Randomize depot
        if cfg.env.randomize_depot:
            depot = (
                random.randint(0, cfg.env.map_size - 1),
                random.randint(0, cfg.env.map_size - 1),
            )
        else:
            depot = cfg.env.depot
        
        # Generate demands
        demands = self.generate_demands(seed=seed)
        
        # Rollout with planner
        planner_reward = self.rollout_with_planner(demands, depot=depot)
        
        # Compute loss
        loss, metrics = self.compute_adversarial_loss(demands, planner_reward)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_MAX_GRAD_NORM)
        self.optimizer.step()

        metrics["loss"] = loss.item()
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        """
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
                print(f"  Episode {ep+1}/{n_episodes}: "
                      f"planner_r={metrics['planner_reward']:.2f}, "
                      f"gen_r={metrics['gen_reward']:.2f}, "
                      f"loss={metrics['loss']:.4f}")
        
        # Average metrics
        avg_metrics = {k: v / n_episodes for k, v in total_metrics.items()}
        return avg_metrics
    
    def save_checkpoint(self, path: str, epoch: int, extra_state: Optional[Dict] = None):
        """Save generator checkpoint."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline": self.baseline,
        }
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, path)
        print(f"[GeneratorTrainer] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load generator checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        if "baseline" in ckpt:
            self.baseline = ckpt["baseline"]
        
        return ckpt.get("epoch", 0)
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict for registry."""
        return self.model.state_dict()
