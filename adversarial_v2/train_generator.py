"""
Generator Trainer Module

Adversarial training for diffusion generator to find planner weaknesses.
Uses RLGeneratorTrainer from training/generator/train_rl_diffusion_generator.py

Supports multi-GPU training with DDP:
- When num_gpus > 1, wraps model with DistributedDataParallel
- Uses gradient accumulation for batch training
- Each GPU processes batch_size/num_gpus episodes per batch
- Gradients are synchronized across GPUs after accumulation
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from agent.generator.data_utils import CONDITION_DIM, prepare_condition
from environment.env import GridEnvironment
from training.generator.train_rl_diffusion_generator import RLGeneratorTrainer
from training.generator.rl_utils import (
    make_environment,
    calculate_spatial_entropy,
    calculate_temporal_entropy,
    normalize_demands_for_training,
)
from configs import get_default_config

# Default hyperparameters
DEFAULT_GEN_LR = 2e-6
DEFAULT_BASELINE_BETA = 0.9
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_GEN_BATCH_SIZE = 8  # Number of episodes to accumulate before update
DEFAULT_ENTROPY_WEIGHT = 0.80
DEFAULT_TIME_ENTROPY_WEIGHT = 0.05
DEFAULT_SL_WEIGHT = 0.1
DEFAULT_DIFF_LOSS_CLIP = 10.0


class GeneratorTrainer:
    """
    Adversarial trainer for diffusion generator with batch training support.
    
    The goal is to train the generator to produce demand distributions
    that maximize difficulty for the planner (minimize planner reward).
    
    Key features:
    1. Batch training via gradient accumulation
    2. Multi-GPU support with DDP
    3. Each GPU processes batch_size/num_gpus episodes per batch
    4. Gradients are accumulated and synchronized across GPUs
    
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
        self.rng = random.Random(config.seed)
        
        # Multi-GPU settings
        self.num_gpus = getattr(config, 'num_gpus', 1)
        self.local_rank = getattr(config, 'local_rank', 0)
        self.distributed = is_distributed()
        
        # Batch settings
        self.batch_size = getattr(config, 'batch_size', DEFAULT_GEN_BATCH_SIZE)
        
        # Training hyperparameters
        self.baseline_beta = DEFAULT_BASELINE_BETA
        self.entropy_weight = DEFAULT_ENTROPY_WEIGHT
        self.time_entropy_weight = DEFAULT_TIME_ENTROPY_WEIGHT
        self.sl_weight = DEFAULT_SL_WEIGHT
        self.diff_loss_clip = DEFAULT_DIFF_LOSS_CLIP
        self.normalize_reward = False
        
        # Baseline for advantage computation (EMA)
        self.baseline = None
        
        # Initialize diffusion model
        self._init_diffusion()
        
        # Wrap model with DDP if distributed
        if self.distributed:
            self.model = wrap_model_ddp(self.model, self.device)
            print_rank0(f"[GeneratorTrainer] Wrapped diffusion model with DDP on rank {self.local_rank}")
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize V2Planner for rollout (wraps the static model)
        self._init_v2_planner()
        
        # Build config object for rollout
        self._build_trainer_cfg()
        
        # Create environment for rollout
        self.env = GridEnvironment(
            width=config.env.map_size,
            height=config.env.map_size,
            num_agents=config.env.num_agents,
            capacity=config.env.capacity,
            depot=config.env.depot,
            max_time=config.env.max_time,
            max_end_time=config.env.max_end_time,
        )
        
        # Prepare default condition
        self.condition = prepare_condition({}).unsqueeze(0).to(self.device)
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=DEFAULT_GEN_LR,
        )
    
    def _build_trainer_cfg(self):
        """Build config dict compatible with decode_demands_from_tensor."""
        cfg = self.config
        
        default_cfg = get_default_config()
        gen_params = default_cfg.generator_params
        
        # Build dict for decode_demands_from_tensor
        self.trainer_cfg = {
            "width": cfg.env.map_size,
            "height": cfg.env.map_size,
            "max_time": cfg.env.max_time,
            "max_c": gen_params.get("max_demand", 5),
            "min_lifetime": gen_params.get("lifetime_min", 10),
            "max_lifetime": gen_params.get("lifetime_max", 50),
        }
        
        # Also keep object-style for normalize_demands_for_training
        class TrainerCfgObj:
            pass
        self.trainer_cfg_obj = TrainerCfgObj()
        self.trainer_cfg_obj.width = cfg.env.map_size
        self.trainer_cfg_obj.height = cfg.env.map_size
        self.trainer_cfg_obj.max_time = cfg.env.max_time
        self.trainer_cfg_obj.depot = cfg.env.depot
        self.trainer_cfg_obj.generator_params = gen_params
        
    
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
        # Unwrap DDP model if necessary
        planner_to_sync = unwrap_model(self.planner_model) if self.distributed else self.planner_model
        self.v2_planner._ensure_model_loaded()
        self.v2_planner._model.load_state_dict(planner_to_sync.state_dict(), strict=False)
        self.v2_planner._model.eval()
    
    def update_planner(self, planner_model: nn.Module):
        """Update the planner reference (after planner training)."""
        self.planner_model = planner_model
        self._sync_planner_weights()
    
    def generate_demands(self, seed: int, depot: Tuple[int, int]) -> List[Tuple[int, int, int, int, int]]:
        """
        Generate demands using diffusion model.
        
        Args:
            seed: Random seed for generation
            depot: Depot location (x, y)
        
        Returns:
            List of demand tuples (x, y, t, c, end_t)
        """
        from training.generator.rl_utils import decode_demands_from_tensor, apply_static_constraints
        
        cfg = self.config
        
        # Update condition with depot
        cond_params = {"depot": depot}
        condition = prepare_condition(cond_params).unsqueeze(0).to(self.device)
        
        # Get underlying model for sampling
        model_for_sample = unwrap_model(self.model) if self.distributed else self.model
        
        with torch.no_grad():
            model_for_sample.eval()
            output = model_for_sample.sample(
                condition=condition,
                num_demands=cfg.env.total_demand,
                grid_size=(cfg.env.map_size, cfg.env.map_size),
            )
        
        # Decode to demands
        demands = decode_demands_from_tensor(output, self.trainer_cfg)
        
        # Apply static constraints if in static mode
        if cfg.mode == "static":
            demands = apply_static_constraints(demands, cfg.env.max_time)
        
        return demands
    
    def rollout(
        self,
        demands: List[Tuple[int, int, int, int, int]],
        depot: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """
        Roll out one episode with the planner.
        
        Returns:
            serviced_reward: Total reward from serviced demands
            total_demand_cap: Total capacity of all demands
            total_failed_cap: Capacity of unserviced + expired demands
        """
        from training.generator.train_rl_diffusion_generator import _plan_episode
        
        self.env.depot = depot
        
        return _plan_episode(
            self.v2_planner,
            self.env,
            demands,
            static_mode=(self.config.mode == "static"),
        )
    
    def compute_loss_for_episode(
        self,
        demands: List[Tuple[int, int, int, int, int]],
        serviced_reward: float,
        total_demand_cap: float,
        total_failed_cap: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a single episode (without backward).
        
        Returns:
            loss: Tensor to accumulate
            metrics: Dict with detailed metrics
        """
        # Generator reward = failed capacity - serviced reward
        gen_reward = total_failed_cap - serviced_reward
        
        # Entropy bonuses
        entropy_bonus = self.entropy_weight * calculate_spatial_entropy(demands)
        time_bonus = self.time_entropy_weight * calculate_temporal_entropy(demands)
        gen_reward += entropy_bonus + time_bonus
        
        # Store gen_reward before penalty
        gen_reward_for_baseline = gen_reward
        
        # Penalty for too few demands
        if total_demand_cap < 10:
            gen_reward -= 100
        
        # Advantage computation
        if self.baseline is None:
            self.baseline = gen_reward_for_baseline
        adv = gen_reward - self.baseline
        
        # Update baseline (EMA)
        self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * gen_reward_for_baseline
        
        # Scale advantage
        if self.normalize_reward:
            adv_scaled = torch.tanh(torch.tensor(
                adv / (abs(self.baseline) + 1e-6),
                dtype=torch.float32, device=self.device
            ))
        else:
            adv_scaled = torch.tensor(adv, dtype=torch.float32, device=self.device)
        
        # Normalize demands for training (uses object-style config)
        x_start = normalize_demands_for_training(demands, self.trainer_cfg_obj).to(self.device).unsqueeze(0)
        
        # Forward diffusion loss
        self.model.train()
        noise, predicted_noise = self.model(x_start, self.condition)
        diff_loss = F.mse_loss(predicted_noise, noise)
        diff_loss_clipped = torch.clamp(diff_loss, max=self.diff_loss_clip)
        
        # Final loss
        loss = diff_loss_clipped * (adv_scaled + self.sl_weight)
        
        metrics = {
            "serviced_reward": serviced_reward,
            "gen_reward": gen_reward,
            "entropy_bonus": entropy_bonus,
            "advantage": adv,
            "diff_loss": diff_loss.item(),
            "loss": loss.item(),
            "baseline": self.baseline,
        }
        
        return loss, metrics
    
    def train_batch(self, batch_seed: int) -> Dict[str, float]:
        """
        Train one batch using gradient accumulation.
        
        In distributed mode:
        - Total batch_size episodes are split across GPUs
        - Each GPU processes local_batch_size = batch_size / num_gpus episodes
        - Gradients are accumulated locally, then DDP syncs on optimizer.step()
        
        Args:
            batch_seed: Base seed for this batch
        
        Returns:
            Aggregated metrics for the batch
        """
        cfg = self.config
        
        # Calculate local batch size for this GPU
        world_size = get_world_size() if self.distributed else 1
        rank = get_rank() if self.distributed else 0
        local_batch_size = self.batch_size // world_size
        if local_batch_size < 1:
            local_batch_size = 1
        
        # Zero gradients at start of batch
        self.optimizer.zero_grad()
        
        # Accumulators for metrics
        total_loss = 0.0
        total_metrics = {}
        
        for i in range(local_batch_size):
            # Each GPU uses different seed offset based on rank
            episode_seed = batch_seed + rank * local_batch_size + i
            
            # Randomize depot
            rng = random.Random(episode_seed)
            if cfg.env.randomize_depot:
                depot = (
                    rng.randint(0, cfg.env.map_size - 1),
                    rng.randint(0, cfg.env.map_size - 1)
                )
            else:
                depot = cfg.env.depot
            
            # Generate demands
            demands = self.generate_demands(seed=episode_seed, depot=depot)
            
            # Rollout
            serviced_reward, total_demand_cap, total_failed_cap = self.rollout(demands, depot)
            
            # Compute loss (but don't step yet)
            loss, metrics = self.compute_loss_for_episode(
                demands, serviced_reward, total_demand_cap, total_failed_cap
            )
            
            # Scale loss by local_batch_size for gradient accumulation
            # This ensures the total gradient magnitude is independent of batch size
            scaled_loss = loss / local_batch_size
            scaled_loss.backward()
            
            total_loss += loss.item()
            
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_MAX_GRAD_NORM)
        
        # Step optimizer - DDP will sync gradients here automatically
        self.optimizer.step()
        
        # Average metrics over local batch
        avg_metrics = {k: v / local_batch_size for k, v in total_metrics.items()}
        avg_metrics["batch_loss"] = total_loss / local_batch_size
        
        return avg_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch using batch training.
        
        In distributed mode:
        - All GPUs process every batch together
        - Each batch's episodes are split across GPUs
        - Gradients are accumulated locally, then synced via DDP
        
        Returns:
            Dictionary of training metrics
        """
        cfg = self.config
        
        # Number of batches per epoch
        n_batches = cfg.episodes_per_epoch // self.batch_size
        if n_batches < 1:
            n_batches = 1
        
        total_metrics = {}
        
        for batch_idx in range(n_batches):
            # Use same base seed for all GPUs to ensure consistent behavior
            batch_seed = cfg.seed + batch_idx * self.batch_size
            
            metrics = self.train_batch(batch_seed)
            
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            
            if (batch_idx + 1) % 10 == 0:
                print_rank0(f"  Batch {batch_idx+1}/{n_batches}: "
                      f"planner_r={metrics.get('serviced_reward', 0):.2f}, "
                      f"gen_r={metrics.get('gen_reward', 0):.2f}, "
                      f"loss={metrics.get('batch_loss', 0):.4f}")
        
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        
        # Rename for compatibility
        return {
            "diff_loss": avg_metrics.get("diff_loss", 0.0),
            "advantage": avg_metrics.get("advantage", 0.0),
            "gen_reward": avg_metrics.get("gen_reward", 0.0),
            "planner_reward": avg_metrics.get("serviced_reward", 0.0),
            "baseline": avg_metrics.get("baseline", 0.0),
            "loss": avg_metrics.get("batch_loss", 0.0),
        }
    
    def save_checkpoint(self, path: str, epoch: int, extra_state: Optional[Dict] = None):
        """Save generator checkpoint. Only saves on main process in distributed mode."""
        # Only save on main process
        if self.distributed and not is_main_process():
            barrier()
            return
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        # Unwrap DDP model to save raw state
        model_to_save = unwrap_model(self.model) if self.distributed else self.model
        
        state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline": self.baseline,
        }
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, path)
        print_rank0(f"[GeneratorTrainer] Saved checkpoint to {path}")
        
        if self.distributed:
            barrier()
    
    def load_checkpoint(self, path: str) -> int:
        """Load generator checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        # Load into underlying model (not DDP wrapper)
        model_to_load = unwrap_model(self.model) if self.distributed else self.model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "baseline" in ckpt:
            self.baseline = ckpt["baseline"]
        
        return ckpt.get("epoch", 0)
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict for registry."""
        model_to_save = unwrap_model(self.model) if self.distributed else self.model
        return model_to_save.state_dict()