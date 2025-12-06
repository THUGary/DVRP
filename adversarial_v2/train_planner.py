"""
Planner Trainer Module

Train V2Planner using distributions from multiple generator versions.
Reuses existing training functions from training_v2/train_static.py and train_dynamic.py

Supports multi-GPU training with DistributedDataParallel (DDP).
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import random
import torch
import torch.nn as nn

from .config import CoevolutionConfig
from .utils.registry import GeneratorRegistry, GeneratorVersion
from .utils.demand_converter import DemandConverter, DemandTuple
from .utils.problem_cache import ProblemCacheManager
from .utils.distributed import (
    is_distributed, is_main_process, get_world_size, get_rank,
    wrap_model_ddp, unwrap_model, reduce_tensor, barrier, print_rank0
)

# Project imports
from configs import COORD_NORM, DEMAND_NORM
from models_v2.static_model import StaticVRPModel, StaticVRPEnv, create_static_model
from models_v2.dynamic_model import DynamicVRPModel, create_dynamic_model
from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import prepare_condition, CONDITION_DIM
from training_v2.train_static import train_one_batch as train_static_one_batch
from training_v2.train_static import generate_random_problems

# Default model hyperparameters (same as training_v2/train_static.py)
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_ENCODER_LAYERS = 6
DEFAULT_HEADS = 8
DEFAULT_QKV_DIM = 16
DEFAULT_FF_HIDDEN = 512
DEFAULT_ADAPTER_DIM = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_POMO_SIZE = 100
DEFAULT_AUG_FACTOR = 1

# Problem cache defaults
DEFAULT_CACHE_REUSE_RATIO = 0.8  # 80% from cache, 20% fresh
DEFAULT_MAX_PROBLEMS_PER_VERSION = 1000
DEFAULT_MIN_CACHE_SIZE_FOR_REUSE = 100


class PlannerTrainer:
    """
    Trainer for V2Planner that uses multi-version generator distributions.
    
    Key features:
    1. Samples from multiple generator versions to avoid policy cycling
    2. Supports both static (POMO) and dynamic (step-by-step) training
    3. REINFORCE with self-competitive baseline
    4. Reuses existing training functions from training_v2
    5. Problem caching to avoid repeated diffusion sampling
    
    Model config uses defaults from training_v2/train_static.py.
    """
    
    def __init__(
        self,
        config: CoevolutionConfig,
        registry: GeneratorRegistry,
        device: torch.device,
        cache_reuse_ratio: Optional[float] = None,
        max_problems_per_version: Optional[int] = None,
        min_cache_size_for_reuse: Optional[int] = None,
    ):
        """
        Initialize planner trainer.
        
        Args:
            config: CoevolutionConfig object containing all settings
            registry: GeneratorRegistry for version sampling
            device: torch device for training
            cache_reuse_ratio: Probability of using cached problems vs fresh generation
                              (0.0 = always generate, 1.0 = always use cache)
                              If None, uses config.cache_reuse_ratio
            max_problems_per_version: Max problems to cache per generator version
                              If None, uses config.max_problems_per_version
            min_cache_size_for_reuse: Minimum cache size before enabling reuse
                              If None, uses config.min_cache_size_for_reuse
        """
        self.config = config
        self.registry = registry
        self.device = device
        self.rng = random.Random(config.seed)
        
        # Multi-GPU settings
        self.num_gpus = getattr(config, 'num_gpus', 1)
        self.local_rank = getattr(config, 'local_rank', 0)
        self.distributed = is_distributed()
        
        # Initialize models
        self._init_planner()
        self._init_diffusion()
        
        # Wrap planner model with DDP if distributed
        if self.distributed:
            self.model = wrap_model_ddp(self.model, self.device)
            print_rank0(f"[PlannerTrainer] Wrapped planner model with DDP on rank {self.local_rank}")
        
        self._init_optimizer()
        
        # Get cache settings from config or use defaults
        _cache_reuse_ratio = cache_reuse_ratio if cache_reuse_ratio is not None \
            else getattr(config, 'cache_reuse_ratio', DEFAULT_CACHE_REUSE_RATIO)
        _max_problems = max_problems_per_version if max_problems_per_version is not None \
            else getattr(config, 'max_problems_per_version', DEFAULT_MAX_PROBLEMS_PER_VERSION)
        _min_cache_size = min_cache_size_for_reuse if min_cache_size_for_reuse is not None \
            else getattr(config, 'min_cache_size_for_reuse', DEFAULT_MIN_CACHE_SIZE_FOR_REUSE)
        
        # Initialize problem cache
        self.problem_cache = ProblemCacheManager(
            cache_dir=os.path.join(config.save_dir, "problem_cache"),
            max_problems_per_version=_max_problems,
            cache_reuse_ratio=_cache_reuse_ratio,
            min_cache_size_for_reuse=_min_cache_size,
        )
        
        print_rank0(f"[PlannerTrainer] Problem cache initialized: "
              f"reuse_ratio={_cache_reuse_ratio:.1%}, "
              f"max_per_version={_max_problems}, "
              f"min_for_reuse={_min_cache_size}")
        
        # Demand converter
        self.converter = DemandConverter(
            map_size=config.env.map_size,
            max_time=config.env.max_time,
            max_end_time=config.env.max_end_time,
            max_c=config.env.max_c,
            min_lifetime=config.env.min_lifetime,
            max_lifetime=config.env.max_lifetime,
        )
        
        # Use num_nodes from config (actual node count for tensor shapes)
        self.num_nodes = config.env.num_nodes
        
        # Get pomo_size from config (defaults to DEFAULT_POMO_SIZE if not set)
        self.pomo_size = getattr(config, 'pomo_size', DEFAULT_POMO_SIZE)
        
        # For static training - reuse StaticVRPEnv
        if config.mode == "static":
            self.static_env = StaticVRPEnv(
                problem_size=self.num_nodes,
                pomo_size=self.pomo_size,
            )
    
    def _init_planner(self):
        """Initialize the planner model using default architecture from train_static.py."""
        cfg = self.config
        
        if cfg.mode == "static":
            # Static mode: Use StaticVRPModel directly for POMO training
            # Architecture defaults from training_v2/train_static.py
            self.model = create_static_model(
                embedding_dim=DEFAULT_EMBEDDING_DIM,
                encoder_layers=DEFAULT_ENCODER_LAYERS,
                heads=DEFAULT_HEADS,
                qkv_dim=DEFAULT_QKV_DIM,
                ff_hidden=DEFAULT_FF_HIDDEN,
            )
            
            # Load checkpoint if provided
            if cfg.planner_checkpoint and os.path.exists(cfg.planner_checkpoint):
                ckpt = torch.load(cfg.planner_checkpoint, map_location=self.device)
                if 'model_state_dict' in ckpt:
                    self.model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(ckpt)
                print_rank0(f"[PlannerTrainer] Loaded static model from {cfg.planner_checkpoint}")
        else:
            # Dynamic mode: Use DynamicVRPModel with adapter
            self.model = create_dynamic_model(
                static_model_or_checkpoint=cfg.planner_checkpoint,
                embedding_dim=DEFAULT_EMBEDDING_DIM,
                encoder_layers=DEFAULT_ENCODER_LAYERS,
                heads=DEFAULT_HEADS,
                qkv_dim=DEFAULT_QKV_DIM,
                ff_hidden=DEFAULT_FF_HIDDEN,
                adapter_dim=DEFAULT_ADAPTER_DIM,
                freeze_static=True,
                device=str(self.device),
            )
        
        self.model = self.model.to(self.device)
    
    def _init_diffusion(self):
        """Initialize diffusion model for demand generation."""
        # Diffusion model defaults from models/generator_model/diffusion_model.py
        self.diffusion_model = DemandDiffusionModel(
            condition_dim=CONDITION_DIM,
            data_dim=5,
            time_emb_dim=64,
            num_steps=1000,
        ).to(self.device)
        
        # Prepare default condition
        self.condition = prepare_condition({}).unsqueeze(0).to(self.device)
    
    def _init_optimizer(self):
        """Initialize optimizer using default hyperparameters from train_static.py."""
        if self.config.mode == "static":
            params = self.model.parameters()
        else:
            # Dynamic mode: only train adapter parameters
            params = self.model.get_trainable_params()
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=DEFAULT_LR,
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )
    
    def load_generator_version(self, version: GeneratorVersion):
        """Load a specific generator version into diffusion model."""
        state_dict = version.load_state_dict(str(self.device))
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        
        self.diffusion_model.load_state_dict(state_dict, strict=False)
        self.diffusion_model.eval()
    
    def load_generator_checkpoint(self, checkpoint_path: str):
        """Load generator directly from checkpoint path (not via registry)."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        
        self.diffusion_model.load_state_dict(state_dict, strict=False)
        self.diffusion_model.eval()
        print_rank0(f"[PlannerTrainer] Loaded generator from {checkpoint_path}")
    
    def generate_problems_from_diffusion(
        self,
        batch_size: int,
        version: Optional[GeneratorVersion] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate VRP problems using diffusion model.
        
        Returns tensors in the format expected by StaticVRPEnv:
            depot_xy: (batch_size, 1, 2) - normalized [0,1]
            node_xy: (batch_size, N, 2) - normalized [0,1]
            node_demand: (batch_size, N) - normalized by DEMAND_NORM
        """
        cfg = self.config
        
        # Load generator version if specified
        if version is not None:
            self.load_generator_version(version)
        
        depot_list = []
        node_xy_list = []
        node_demand_list = []
        
        for _ in range(batch_size):
            # Randomize depot if configured
            if cfg.env.randomize_depot:
                depot = (
                    self.rng.random(),  # Already in [0,1]
                    self.rng.random(),
                )
            else:
                depot = (cfg.env.depot[0] / cfg.env.map_size, cfg.env.depot[1] / cfg.env.map_size)
            
            # Generate demands using diffusion (DDIM for speed)
            with torch.no_grad():
                if hasattr(self.diffusion_model, 'sample_ddim'):
                    output = self.diffusion_model.sample_ddim(
                        condition=self.condition,
                        num_demands=cfg.env.total_demand,
                        grid_size=(cfg.env.map_size, cfg.env.map_size),
                        num_inference_steps=50,  # 50 steps vs 1000 for DDPM
                        eta=0.0,
                    )
                else:
                    output = self.diffusion_model.sample(
                        condition=self.condition,
                        num_demands=cfg.env.total_demand,
                        grid_size=(cfg.env.map_size, cfg.env.map_size),
                    )
            
            # Convert to static demands
            demands = self.converter.convert(output, mode=cfg.mode)
            
            # Build tensors in normalized format [0,1]
            depot_xy = torch.tensor([[[depot[0], depot[1]]]], dtype=torch.float32)
            
            # Normalize coordinates to [0,1]
            node_coords = [[d.x / cfg.env.map_size, d.y / cfg.env.map_size] for d in demands]
            node_xy = torch.tensor([node_coords], dtype=torch.float32)
            
            # Normalize demands by DEMAND_NORM (same as train_static.py)
            node_demand = torch.tensor([[d.c / DEMAND_NORM for d in demands]], 
                                      dtype=torch.float32)
            
            depot_list.append(depot_xy)
            node_xy_list.append(node_xy)
            node_demand_list.append(node_demand)
        
        # Stack batches
        depot_xy = torch.cat(depot_list, dim=0).to(self.device)
        node_xy = torch.cat(node_xy_list, dim=0).to(self.device)
        node_demand = torch.cat(node_demand_list, dim=0).to(self.device)
        
        return depot_xy, node_xy, node_demand
    
    def get_problems_with_cache(
        self,
        batch_size: int,
        version: GeneratorVersion,
        is_latest_version: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Get problems using cache + fresh generation mix.
        
        Decision logic:
        1. Latest version: ALWAYS generate fresh problems (for adversarial challenge),
           but cache them for future use when this version becomes historical
        2. Historical versions: Use cache if available AND random < cache_reuse_ratio
        3. Otherwise: generate fresh problems and add to cache
        
        Args:
            batch_size: Number of problems needed
            version: Generator version to use
            is_latest_version: If True, always generate fresh (don't read from cache)
        
        Returns:
            depot_xy, node_xy, node_demand, stats dict with cache/fresh counts
        """
        version_id = version.version_id
        stats = {"from_cache": 0, "freshly_generated": 0}
        
        # Latest version: always generate fresh problems for adversarial training
        # This ensures the planner always trains on the latest challenge
        # But still cache them for future use (when this version becomes historical)
        if is_latest_version:
            depot_xy, node_xy, node_demand = self.generate_problems_from_diffusion(
                batch_size, version
            )
            stats["freshly_generated"] = batch_size
            # Cache for future use when this version becomes historical
            self.problem_cache.add_problems(version_id, depot_xy, node_xy, node_demand)
            return depot_xy, node_xy, node_demand, stats
        
        # Historical versions: check if we should use cache
        if self.problem_cache.should_use_cache(version_id, self.rng):
            # Sample from cache
            result = self.problem_cache.sample_from_cache(
                version_id, batch_size, self.rng, self.device
            )
            if result is not None:
                stats["from_cache"] = batch_size
                return result[0], result[1], result[2], stats
        
        # Generate fresh problems for historical version
        depot_xy, node_xy, node_demand = self.generate_problems_from_diffusion(
            batch_size, version
        )
        stats["freshly_generated"] = batch_size
        
        # Add to cache (historical versions are cached for reuse)
        self.problem_cache.add_problems(version_id, depot_xy, node_xy, node_demand)
        
        return depot_xy, node_xy, node_demand, stats
    
    def train_static_batch_with_diffusion(
        self,
        version: Optional[GeneratorVersion] = None,
        use_cache: bool = True,
        is_latest_version: bool = False,
    ) -> Tuple[float, float, Dict[str, int]]:
        """
        Train one batch using POMO-style training with diffusion-generated problems.
        
        In distributed mode, each GPU processes batch_size/num_gpus samples,
        and DDP automatically synchronizes gradients across GPUs.
        
        Args:
            version: Generator version to use (loads into diffusion model)
            use_cache: Whether to use problem cache (default True)
            is_latest_version: If True, always generate fresh problems (don't use cache)
        
        Returns:
            avg_score: average tour length (lower is better)
            loss: policy gradient loss
            cache_stats: dict with cache usage statistics
        """
        cfg = self.config
        cache_stats = {"from_cache": 0, "freshly_generated": 0}
        
        # In distributed mode, each GPU handles a portion of the batch
        # Total batch_size is split across GPUs
        world_size = get_world_size() if self.distributed else 1
        local_batch_size = cfg.batch_size // world_size
        
        # Ensure at least 1 sample per GPU
        if local_batch_size < 1:
            local_batch_size = 1
        
        # Get problems (with cache if enabled and version provided)
        # Each GPU generates/loads its own local_batch_size problems
        if use_cache and version is not None:
            depot_xy, node_xy, node_demand, cache_stats = self.get_problems_with_cache(
                local_batch_size, version, is_latest_version=is_latest_version
            )
        else:
            # Generate problems directly (no cache)
            depot_xy, node_xy, node_demand = self.generate_problems_from_diffusion(
                local_batch_size, version
            )
            cache_stats["freshly_generated"] = local_batch_size
        
        # Use the existing train_one_batch logic
        self.model.train()
        
        pomo_size = self.pomo_size
        aug_factor = DEFAULT_AUG_FACTOR
        
        # Load problems into environment
        self.static_env.load_problems(depot_xy, node_xy, node_demand, aug_factor=aug_factor)
        
        # Reset
        reset_state, _, _ = self.static_env.reset()
        
        # `self.model` may be a DistributedDataParallel wrapper. `pre_forward` is
        # a custom method on the underlying module, not on DDP. Call it on the
        # unwrapped module to avoid AttributeError while keeping forward passes
        # through the DDP wrapper for correct gradient syncing.
        model_for_state = unwrap_model(self.model) if self.distributed else self.model
        model_for_state.pre_forward(reset_state)
        
        # Collect rollout (same as train_static.py)
        prob_list = []
        state, _, done = self.static_env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.static_env.step(selected)
            prob_list.append(prob)
        
        # Stack probabilities
        prob_tensor = torch.stack(prob_list, dim=2)  # (batch, pomo, steps)
        
        # Compute loss (REINFORCE with POMO baseline) - same as train_static.py
        advantage = reward - reward.mean(dim=1, keepdim=True)
        log_prob = prob_tensor.log().sum(dim=2)
        
        loss = -(advantage * log_prob).mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_MAX_GRAD_NORM)
        self.optimizer.step()
        
        # Best score across POMO instances
        best_reward, _ = reward.max(dim=1)
        avg_score = -best_reward.mean().item()
        
        return avg_score, loss.item(), cache_stats
    
    def train_static_batch_random(self) -> Tuple[float, float]:
        """
        Train one batch using random problems.
        
        In distributed mode, each GPU generates and processes batch_size/num_gpus samples.
        
        This is useful for baseline comparison or warm-up.
        """
        cfg = self.config
        
        # In distributed mode, each GPU handles a portion of the batch
        world_size = get_world_size() if self.distributed else 1
        local_batch_size = cfg.batch_size // world_size
        if local_batch_size < 1:
            local_batch_size = 1
        
        # Generate random problems for this GPU's portion
        depot_xy, node_xy, node_demand = generate_random_problems(
            batch_size=local_batch_size,
            problem_size=self.num_nodes,
            device=self.device,
            target_num_vehicles=cfg.env.num_agents,
        )
        
        # Use the same training logic as diffusion-generated problems
        self.model.train()
        
        pomo_size = self.pomo_size
        aug_factor = DEFAULT_AUG_FACTOR
        
        # Load problems into environment
        self.static_env.load_problems(depot_xy, node_xy, node_demand, aug_factor=aug_factor)
        
        # Reset
        reset_state, _, _ = self.static_env.reset()
        
        model_for_state = unwrap_model(self.model) if self.distributed else self.model
        model_for_state.pre_forward(reset_state)
        
        # Collect rollout
        prob_list = []
        state, _, done = self.static_env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.static_env.step(selected)
            prob_list.append(prob)
        
        # Stack probabilities
        prob_tensor = torch.stack(prob_list, dim=2)
        
        # Compute loss (REINFORCE with POMO baseline)
        advantage = reward - reward.mean(dim=1, keepdim=True)
        log_prob = prob_tensor.log().sum(dim=2)
        
        loss = -(advantage * log_prob).mean()
        
        # Backward - DDP will sync gradients automatically
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_MAX_GRAD_NORM)
        self.optimizer.step()
        
        # Best score across POMO instances
        best_reward, _ = reward.max(dim=1)
        avg_score = -best_reward.mean().item()
        
        return avg_score, loss.item()
    
    def train_epoch(self, use_diffusion: bool = True) -> Dict[str, float]:
        """
        Train for one epoch, sampling from multiple generator versions.
        
        In distributed mode:
        - All GPUs process every batch together
        - Each batch's data (batch_size samples) is split across GPUs
        - Each GPU processes batch_size/num_gpus samples
        - DDP automatically synchronizes gradients after each batch
        
        Args:
            use_diffusion: If True, use diffusion-generated problems.
                          If False, use random problems (baseline).
        
        Returns:
            Dictionary of training metrics
        """
        cfg = self.config
        n_batches = cfg.episodes_per_epoch // cfg.batch_size

        if n_batches == 0:
            return {
                "score": 0.0,
                "loss": 0.0,
                "version_counts": {},
                "cache_stats": {
                    "from_cache": 0,
                    "freshly_generated": 0,
                    "cache_hit_rate": 0.0,
                    "total_cached_problems": self.problem_cache.get_cache_stats()["total_problems"],
                },
            }

        # All GPUs process every batch together
        # Each GPU handles batch_size/num_gpus samples per batch
        total_score = 0.0
        total_loss = 0.0
        version_counts: Dict[int, int] = {}
        total_from_cache = 0
        total_freshly_generated = 0

        # Get latest version ID for cache decision
        latest_version_id = self.registry.latest().version_id if not self.registry.is_empty() else -1

        for batch_idx in range(n_batches):
            if use_diffusion and not self.registry.is_empty():
                # Sample generator version (same version for all GPUs in this batch)
                # Use batch_idx as seed offset to ensure all GPUs sample same version
                batch_rng = random.Random(cfg.seed + batch_idx)
                
                if cfg.version_sample_policy == "all":
                    version = self.registry.all_versions()[batch_idx % self.registry.num_versions()]
                else:
                    version = self.registry.sample(
                        policy=cfg.version_sample_policy,
                        latest_bias=cfg.latest_bias,
                        rng=batch_rng,
                    )
                version_id = version.version_id
                version_counts[version_id] = version_counts.get(version_id, 0) + 1

                is_latest = (version_id == latest_version_id)

                # Each GPU generates and processes its portion of the batch
                # DDP handles gradient synchronization automatically
                score, loss, cache_stats = self.train_static_batch_with_diffusion(
                    version, is_latest_version=is_latest
                )
                total_from_cache += cache_stats.get("from_cache", 0)
                total_freshly_generated += cache_stats.get("freshly_generated", 0)
            else:
                version_id = 0
                score, loss = self.train_static_batch_random()

            total_score += score
            total_loss += loss

            # Periodic logging (only main process)
            if (batch_idx + 1) % 10 == 0:
                cache_rate = total_from_cache / max(1, total_from_cache + total_freshly_generated) * 100
                is_latest_str = " [LATEST]" if (use_diffusion and not self.registry.is_empty() and version_id == latest_version_id) else ""
                print_rank0(f"  Batch {batch_idx+1}/{n_batches}: score={score:.4f}, loss={loss:.4f}, "
                      f"gen_v{version_id}{is_latest_str}, cache_rate={cache_rate:.0f}%")

        avg_score = total_score / n_batches
        avg_loss = total_loss / n_batches

        # Log cache statistics
        cache_stats_summary = self.problem_cache.get_cache_stats()

        return {
            "score": avg_score,
            "loss": avg_loss,
            "version_counts": version_counts,
            "cache_stats": {
                "from_cache": total_from_cache,
                "freshly_generated": total_freshly_generated,
                "cache_hit_rate": total_from_cache / max(1, total_from_cache + total_freshly_generated),
                "total_cached_problems": cache_stats_summary["total_problems"],
            },
        }
    
    def save_checkpoint(self, path: str, epoch: int, extra_state: Optional[Dict] = None):
        """Save planner checkpoint. Only saves on main process in distributed mode."""
        # Only save on main process
        if self.distributed and not is_main_process():
            # Sync before returning to ensure all processes finish together
            barrier()
            return
            
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        state = {
            "epoch": epoch,
            "mode": self.config.mode,
        }
        
        # Unwrap DDP model to save raw model state
        model_to_save = unwrap_model(self.model) if self.distributed else self.model
        
        if self.config.mode == "static":
            state["model_state_dict"] = model_to_save.state_dict()
        else:
            state["adapter_state"] = model_to_save.adapter_state_dict()
        
        state["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, path)
        print_rank0(f"[PlannerTrainer] Saved checkpoint to {path}")
        
        # Also save problem cache to disk
        self.problem_cache.save_all()
        
        # Sync after saving
        if self.distributed:
            barrier()
    
    def load_checkpoint(self, path: str):
        """Load planner checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        if self.config.mode == "static":
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_adapter_state_dict(ckpt["adapter_state"])
        
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        return ckpt.get("epoch", 0)
    
    def populate_cache_for_version(
        self,
        version: GeneratorVersion,
        num_problems: int,
        batch_size: int = 32,
    ):
        """
        Pre-populate cache for a specific generator version.
        
        Useful for warming up the cache before training.
        
        Args:
            version: Generator version to generate problems from
            num_problems: Number of problems to generate and cache
            batch_size: Batch size for generation
        """
        print_rank0(f"[PlannerTrainer] Populating cache for v{version.version_id} "
              f"with {num_problems} problems...")
        
        self.load_generator_version(version)
        
        generated = 0
        while generated < num_problems:
            n = min(batch_size, num_problems - generated)
            depot_xy, node_xy, node_demand = self.generate_problems_from_diffusion(n, version)
            self.problem_cache.add_problems(version.version_id, depot_xy, node_xy, node_demand)
            generated += n
            if generated % 100 == 0:
                print_rank0(f"  Generated {generated}/{num_problems}")
        
        # Save to disk
        self.problem_cache.save_all()
        print_rank0(f"[PlannerTrainer] Cache populated: {self.problem_cache.get_cache_stats()}")
