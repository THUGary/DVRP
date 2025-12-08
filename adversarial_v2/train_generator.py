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
        
        # Initialize static environment for batch rollout (optimization for static VRP)
        self._init_static_env()
        
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
        )
        
        # Prepare default condition using merged generator params (with 'param_' prefix)
        gen_params = getattr(self, 'trainer_cfg_obj', None)
        if gen_params is not None and hasattr(self.trainer_cfg_obj, 'generator_params'):
            cond_dict = {f"param_{k}": v for k, v in self.trainer_cfg_obj.generator_params.items()}
        else:
            cond_dict = {}
        self.condition = prepare_condition(cond_dict).unsqueeze(0).to(self.device)
    
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
        # Merge defaults with environment overrides (total_demand, max_c)
        env_params = {}
        if hasattr(cfg, 'env') and cfg.env is not None:
            if hasattr(cfg.env, 'total_demand'):
                env_params['total_demand'] = cfg.env.total_demand
            if hasattr(cfg.env, 'max_c'):
                env_params['max_c'] = cfg.env.max_c

        merged_params = {**gen_params, **env_params}

        self.trainer_cfg = {
            "width": cfg.env.map_size,
            "height": cfg.env.map_size,
            "max_time": cfg.env.max_time,
            "max_c": merged_params.get("max_c", gen_params.get("max_demand", 5)),
            "min_lifetime": merged_params.get("min_lifetime", gen_params.get("lifetime_min", 10)),
            "max_lifetime": merged_params.get("max_lifetime", gen_params.get("lifetime_max", 50)),
        }
        
        # Also keep object-style for normalize_demands_for_training
        class TrainerCfgObj:
            pass
        self.trainer_cfg_obj = TrainerCfgObj()
        self.trainer_cfg_obj.width = cfg.env.map_size
        self.trainer_cfg_obj.height = cfg.env.map_size
        self.trainer_cfg_obj.max_time = cfg.env.max_time
        self.trainer_cfg_obj.depot = cfg.env.depot
        # Expose merged generator params to downstream trainers
        self.trainer_cfg_obj.generator_params = merged_params
        
    
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
    
    def _init_static_env(self):
        """Initialize StaticVRPEnv for batch rollout (used in static mode)."""
        from models_v2.static_model import StaticVRPEnv
        from configs import DEMAND_NORM
        
        cfg = self.config
        
        # Create static environment for batch evaluation
        # This allows parallel evaluation of multiple problems
        self.static_env = StaticVRPEnv(
            problem_size=cfg.env.num_nodes,
            pomo_size=1,  # We use pomo_size=1 for generator eval (single route per problem)
            vehicle_capacity=cfg.env.capacity / DEMAND_NORM,  # Normalized capacity
        )
        # Note: StaticVRPEnv is not an nn.Module, tensors are created on device in load_problems
        self.coord_norm = cfg.env.map_size
        self.demand_norm = DEMAND_NORM
    
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
        
        # Update condition with generator params (depot is not part of condition)
        if hasattr(self, 'trainer_cfg_obj') and hasattr(self.trainer_cfg_obj, 'generator_params'):
            cond_dict = {f"param_{k}": v for k, v in self.trainer_cfg_obj.generator_params.items()}
        else:
            cond_dict = {}
        condition = prepare_condition(cond_dict).unsqueeze(0).to(self.device)

        # Get underlying model for sampling
        model_for_sample = unwrap_model(self.model) if self.distributed else self.model
        
        with torch.no_grad():
            model_for_sample.eval()
            # Use DDIM for faster sampling (50 steps instead of 1000)
            # This gives ~20x speedup with minimal quality loss
            if hasattr(model_for_sample, 'sample_ddim'):
                output = model_for_sample.sample_ddim(
                    condition=condition,
                    num_demands=cfg.env.total_demand,
                    grid_size=(cfg.env.map_size, cfg.env.map_size),
                    num_inference_steps=50,  # 50 steps vs 1000 for DDPM
                    eta=0.0,  # Deterministic sampling
                )
            else:
                # Fallback to DDPM
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
    
    def generate_batch_demands(
        self,
        batch_size: int,
        base_seed: int,
    ) -> Tuple[List[List[Tuple[int, int, int, int, int]]], List[Tuple[int, int]]]:
        """
        Generate demands for a batch of episodes using DDIM sampling.
        
        Args:
            batch_size: Number of episodes to generate
            base_seed: Base seed for reproducibility
            
        Returns:
            demands_list: List of demand lists for each episode
            depots: List of depot locations for each episode
        """
        from training.generator.rl_utils import decode_demands_from_tensor, apply_static_constraints
        
        cfg = self.config
        demands_list = []
        depots = []
        
        # Get underlying model for sampling
        model_for_sample = unwrap_model(self.model) if self.distributed else self.model
        
        for i in range(batch_size):
            episode_seed = base_seed + i
            rng = random.Random(episode_seed)
            
            # Randomize depot
            if cfg.env.randomize_depot:
                depot = (
                    rng.randint(0, cfg.env.map_size - 1),
                    rng.randint(0, cfg.env.map_size - 1)
                )
            else:
                depot = cfg.env.depot
            depots.append(depot)
            
            # Generate demands with DDIM
            # cond_params = {"depot": depot}
            if hasattr(self, 'trainer_cfg_obj') and hasattr(self.trainer_cfg_obj, 'generator_params'):
                cond_dict = {f"param_{k}": v for k, v in self.trainer_cfg_obj.generator_params.items()}
            else:
                cond_dict = {}
            condition = prepare_condition(cond_dict).unsqueeze(0).to(self.device)

            with torch.no_grad():
                model_for_sample.eval()
                if hasattr(model_for_sample, 'sample_ddim'):
                    output = model_for_sample.sample_ddim(
                        condition=condition,
                        num_demands=cfg.env.total_demand,
                        grid_size=(cfg.env.map_size, cfg.env.map_size),
                        num_inference_steps=50,
                        eta=0.0,
                    )
                else:
                    output = model_for_sample.sample(
                        condition=condition,
                        num_demands=cfg.env.total_demand,
                        grid_size=(cfg.env.map_size, cfg.env.map_size),
                    )
            
            demands = decode_demands_from_tensor(output, self.trainer_cfg)
            if cfg.mode == "static":
                demands = apply_static_constraints(demands, cfg.env.max_time)
            demands_list.append(demands)
        
        return demands_list, depots
    
    def batch_rollout_static(
        self,
        demands_list: List[List[Tuple[int, int, int, int, int]]],
        depots: List[Tuple[int, int]],
    ) -> List[Tuple[float, float, float]]:
        """
        Batch rollout for static VRP using the POMO model directly.
        
        This is much faster than serial rollouts because:
        1. Single forward pass through the model for all problems
        2. Parallelized on GPU
        
        Args:
            demands_list: List of demand lists for each episode
            depots: List of depot locations for each episode
            
        Returns:
            List of (serviced_reward, total_demand_cap, total_failed_cap) tuples
        """
        batch_size = len(demands_list)
        cfg = self.config
        
        # Prepare tensors for batch processing
        depot_xy_list = []
        node_xy_list = []
        node_demand_list = []
        total_caps = []
        
        for i, (demands, depot) in enumerate(zip(demands_list, depots)):
            # Normalize depot coordinates
            depot_norm = [depot[0] / self.coord_norm, depot[1] / self.coord_norm]
            depot_xy_list.append([[depot_norm[0], depot_norm[1]]])
            
            # Normalize node coordinates and demands
            node_coords = [[d[0] / self.coord_norm, d[1] / self.coord_norm] for d in demands]
            node_demands = [d[3] / self.demand_norm for d in demands]
            
            node_xy_list.append(node_coords)
            node_demand_list.append(node_demands)
            total_caps.append(sum(d[3] for d in demands))
        
        # Stack into tensors
        depot_xy = torch.tensor(depot_xy_list, dtype=torch.float32, device=self.device)
        node_xy = torch.tensor(node_xy_list, dtype=torch.float32, device=self.device)
        node_demand = torch.tensor(node_demand_list, dtype=torch.float32, device=self.device)
        
        # Load problems into static environment
        self.static_env.load_problems(depot_xy, node_xy, node_demand, aug_factor=1)
        
        # Get the planner model (unwrap if DDP)
        planner_model = unwrap_model(self.planner_model) if self.distributed else self.planner_model
        planner_model.eval()
        
        # Run batch rollout
        with torch.no_grad():
            reset_state, _, _ = self.static_env.reset()
            planner_model.pre_forward(reset_state)
            
            state, _, done = self.static_env.pre_step()
            
            while not done:
                selected, _ = planner_model(state)
                state, reward, done = self.static_env.step(selected)
        
        # reward shape: (batch, pomo_size) = (batch, 1) with pomo_size=1
        # reward is negative tour length, so positive means shorter tours
        # We convert to "serviced reward" which should be higher for better solutions
        tour_lengths = -reward.squeeze(1)  # (batch,)
        
        # For static VRP with batch rollout:
        # - tour_length is the total travel distance (normalized coordinates)
        # - We need to return values compatible with compute_loss_for_episode
        # 
        # In the original serial rollout (_plan_episode):
        # - serviced_reward = sum of capacities of serviced demands
        # - total_demand_cap = sum of all demand capacities
        # - failed_cap = capacity of unserviced demands
        #
        # For static VRP (all demands serviced):
        # - serviced_reward ≈ total_demand_cap (all serviced)
        # - failed_cap = 0
        #
        # But we want the generator to maximize tour_length (harder problems)
        # So we use tour_length directly as a proxy:
        # - Higher tour_length = harder problem = better for generator
        # - We set serviced_reward = -tour_length * scale (lower is better for generator)
        
        results = []
        for i in range(batch_size):
            total_cap = total_caps[i]
            tour_len = tour_lengths[i].item()
            
            # Scale tour_length to unnormalized coordinates (for consistent scale)
            # tour_len is in [0,1] normalized space, scale back to grid space
            tour_len_scaled = tour_len * self.coord_norm
            
            # Use tour_length as the "cost" that generator wants to maximize
            # serviced_reward = -tour_length (lower reward for planner = higher for generator)
            # This is consistent with the adversarial objective
            serviced_reward = -tour_len_scaled
            failed_cap = 0.0  # In static VRP, no demands fail
            
            results.append((serviced_reward, float(total_cap), failed_cap))
        
        return results
    
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
        Train one batch using gradient accumulation with batch rollout.
        
        In distributed mode:
        - Total batch_size episodes are split across GPUs
        - Each GPU processes local_batch_size = batch_size / num_gpus episodes
        - Use model.no_sync() for gradient accumulation to avoid redundant all-reduce
        - Only sync gradients on the last backward call, then optimizer.step()
        
        For static VRP, uses batch_rollout_static for parallel evaluation.
        
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
        
        # Compute base seed for this GPU's portion
        gpu_base_seed = batch_seed + rank * local_batch_size
        
        # Generate all demands and depots for this batch
        demands_list, depots = self.generate_batch_demands(local_batch_size, gpu_base_seed)
        
        # Batch rollout (parallel evaluation)
        if cfg.mode == "static":
            rollout_results = self.batch_rollout_static(demands_list, depots)
        else:
            # Fallback to serial rollout for dynamic mode
            rollout_results = []
            for demands, depot in zip(demands_list, depots):
                result = self.rollout(demands, depot)
                rollout_results.append(result)
        
        # Zero gradients at start of batch
        self.optimizer.zero_grad()
        
        # Accumulators for metrics
        total_loss = 0.0
        total_metrics = {}
        
        # For DDP with gradient accumulation:
        # - Use no_sync() for all but the last backward to avoid redundant all-reduce
        # - This ensures gradients are only synced once after all local accumulation
        from contextlib import nullcontext
        
        # Compute loss for each episode and accumulate gradients
        for i, (demands, (serviced_reward, total_demand_cap, total_failed_cap)) in enumerate(
            zip(demands_list, rollout_results)
        ):
            is_last_step = (i == local_batch_size - 1)
            
            # Use no_sync for all steps except the last one in distributed mode
            # This prevents redundant gradient synchronization during accumulation
            if self.distributed and not is_last_step:
                sync_context = self.model.no_sync()
            else:
                sync_context = nullcontext()
            
            with sync_context:
                # Compute loss (but don't step yet)
                loss, metrics = self.compute_loss_for_episode(
                    demands, serviced_reward, total_demand_cap, total_failed_cap
                )
                
                # Scale loss by local_batch_size for gradient accumulation
                # Also scale by world_size to ensure global batch average
                scaled_loss = loss / (local_batch_size * world_size)
                scaled_loss.backward()
            
            total_loss += loss.item()
            
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
        
        # Clip gradients (after all-reduce is complete)
        nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_MAX_GRAD_NORM)
        
        # Step optimizer - at this point all GPUs have synchronized gradients
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
        
        # Build result dict
        result = {
            "diff_loss": avg_metrics.get("diff_loss", 0.0),
            "advantage": avg_metrics.get("advantage", 0.0),
            "gen_reward": avg_metrics.get("gen_reward", 0.0),
            "planner_reward": avg_metrics.get("serviced_reward", 0.0),
            "baseline": avg_metrics.get("baseline", 0.0),
            "loss": avg_metrics.get("batch_loss", 0.0),
        }
        
        # In distributed mode, reduce metrics across all GPUs to get global averages.
        # This is CRITICAL: all GPUs must see the same metric values to make
        # consistent decisions about early stopping and best checkpoint saving.
        if self.distributed:
            for key in result:
                tensor_val = torch.tensor(result[key], device=self.device)
                reduced_val = reduce_tensor(tensor_val, op="mean")
                result[key] = reduced_val.item()
        
        return result
    
    def save_checkpoint(self, path: str, epoch: int, extra_state: Optional[Dict] = None):
        """Save generator checkpoint. Only saves on main process in distributed mode."""
        # Only main process saves
        if is_main_process():
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
        
        # All processes sync here after saving is complete
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