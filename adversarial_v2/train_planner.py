"""
Planner Trainer Module

Train V2Planner using distributions from multiple generator versions.
Reuses existing training functions from training_v2/train_static.py and train_dynamic.py
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


class PlannerTrainer:
    """
    Trainer for V2Planner that uses multi-version generator distributions.
    
    Key features:
    1. Samples from multiple generator versions to avoid policy cycling
    2. Supports both static (POMO) and dynamic (step-by-step) training
    3. REINFORCE with self-competitive baseline
    4. Reuses existing training functions from training_v2
    
    Model config uses defaults from training_v2/train_static.py.
    """
    
    def __init__(
        self,
        config: CoevolutionConfig,
        registry: GeneratorRegistry,
        device: torch.device,
    ):
        """
        Initialize planner trainer.
        
        Args:
            config: CoevolutionConfig object containing all settings
            registry: GeneratorRegistry for version sampling
            device: torch device for training
        """
        self.config = config
        self.registry = registry
        self.device = device
        self.rng = random.Random(config.seed)
        
        # Initialize models
        self._init_planner()
        self._init_diffusion()
        self._init_optimizer()
        
        # Demand converter
        self.converter = DemandConverter(
            map_size=config.env.map_size,
            max_time=config.env.max_time,
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
                print(f"[PlannerTrainer] Loaded static model from {cfg.planner_checkpoint}")
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
            
            # Generate demands using diffusion
            with torch.no_grad():
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
    
    def train_static_batch_with_diffusion(
        self,
        version: Optional[GeneratorVersion] = None,
    ) -> Tuple[float, float]:
        """
        Train one batch using POMO-style training with diffusion-generated problems.
        
        This reuses the logic from training_v2/train_static.py but with
        problems generated by the diffusion model instead of random.
        
        Returns:
            avg_score: average tour length (lower is better)
            loss: policy gradient loss
        """
        cfg = self.config
        
        # Generate problems from diffusion model
        depot_xy, node_xy, node_demand = self.generate_problems_from_diffusion(
            cfg.batch_size, version
        )
        
        # Use the existing train_one_batch logic
        self.model.train()
        
        pomo_size = self.pomo_size
        aug_factor = DEFAULT_AUG_FACTOR
        
        # Load problems into environment
        self.static_env.load_problems(depot_xy, node_xy, node_demand, aug_factor=aug_factor)
        
        # Reset
        reset_state, _, _ = self.static_env.reset()
        self.model.pre_forward(reset_state)
        
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
        
        return avg_score, loss.item()
    
    def train_static_batch_random(self) -> Tuple[float, float]:
        """
        Train one batch using random problems (reuse train_static.py directly).
        
        This is useful for baseline comparison or warm-up.
        """
        cfg = self.config
        
        # Directly reuse train_one_batch from train_static.py
        return train_static_one_batch(
            model=self.model,
            env=self.static_env,
            optimizer=self.optimizer,
            batch_size=cfg.batch_size,
            problem_size=self.num_nodes,
            pomo_size=self.pomo_size,
            aug_factor=DEFAULT_AUG_FACTOR,
            device=self.device,
            target_num_vehicles=cfg.env.num_agents,
        )
    
    def train_epoch(self, use_diffusion: bool = True) -> Dict[str, float]:
        """
        Train for one epoch, sampling from multiple generator versions.
        
        Args:
            use_diffusion: If True, use diffusion-generated problems.
                          If False, use random problems (baseline).
        
        Returns:
            Dictionary of training metrics
        """
        cfg = self.config
        n_batches = cfg.episodes_per_epoch // cfg.batch_size
        
        total_score = 0.0
        total_loss = 0.0
        version_counts = {}
        
        for batch_idx in range(n_batches):
            if use_diffusion and not self.registry.is_empty():
                # Sample generator version
                if cfg.version_sample_policy == "all":
                    # Round-robin through all versions
                    version = self.registry.all_versions()[batch_idx % self.registry.num_versions()]
                else:
                    version = self.registry.sample(
                        policy=cfg.version_sample_policy,
                        latest_bias=cfg.latest_bias,
                        rng=self.rng,
                    )
                version_id = version.version_id
                version_counts[version_id] = version_counts.get(version_id, 0) + 1
                
                # Train with diffusion-generated problems
                score, loss = self.train_static_batch_with_diffusion(version)
            else:
                # Train with random problems (reuse train_static.py)
                version_id = 0
                score, loss = self.train_static_batch_random()
            
            total_score += score
            total_loss += loss
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{n_batches}: score={score:.4f}, loss={loss:.4f}, gen_v{version_id}")
        
        avg_score = total_score / n_batches
        avg_loss = total_loss / n_batches
        
        return {
            "score": avg_score,
            "loss": avg_loss,
            "version_counts": version_counts,
        }
    
    def save_checkpoint(self, path: str, epoch: int, extra_state: Optional[Dict] = None):
        """Save planner checkpoint."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        state = {
            "epoch": epoch,
            "mode": self.config.mode,
        }
        
        if self.config.mode == "static":
            state["model_state_dict"] = self.model.state_dict()
        else:
            state["adapter_state"] = self.model.adapter_state_dict()
        
        state["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, path)
        print(f"[PlannerTrainer] Saved checkpoint to {path}")
    
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
