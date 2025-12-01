"""
Generator Version Registry

Tracks all generator checkpoints to enable sampling from multiple versions
during planner training, preventing policy cycling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import time
import json
import torch
import random


@dataclass
class GeneratorVersion:
    """A single generator version/checkpoint."""
    version_id: int
    checkpoint_path: str
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def load_state_dict(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Load the generator state dict from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        return torch.load(self.checkpoint_path, map_location=device)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "version_id": self.version_id,
            "checkpoint_path": self.checkpoint_path,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorVersion":
        """Create from dict."""
        return cls(
            version_id=data["version_id"],
            checkpoint_path=data["checkpoint_path"],
            timestamp=data.get("timestamp", time.time()),
            metrics=data.get("metrics", {}),
        )


class GeneratorRegistry:
    """
    Registry to track all generator versions.
    
    This enables:
    1. Sampling from multiple versions during planner training (avoid policy cycling)
    2. Tracking evolution of generator performance over time
    3. Persistence of version history across runs
    """
    
    def __init__(self, save_dir: str = "checkpoints/adversarial_v2"):
        self.save_dir = save_dir
        self._versions: List[GeneratorVersion] = []
        self._next_id = 1
        self._registry_path = os.path.join(save_dir, "generator_registry.json")
        
    def add(
        self, 
        checkpoint_path: str, 
        metrics: Optional[Dict[str, Any]] = None
    ) -> GeneratorVersion:
        """Add a new generator version to the registry."""
        version = GeneratorVersion(
            version_id=self._next_id,
            checkpoint_path=checkpoint_path,
            metrics=metrics or {},
        )
        self._versions.append(version)
        self._next_id += 1
        return version
    
    def get(self, version_id: int) -> Optional[GeneratorVersion]:
        """Get a specific version by ID."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None
    
    def latest(self) -> Optional[GeneratorVersion]:
        """Get the most recent version."""
        return self._versions[-1] if self._versions else None
    
    def all_versions(self) -> List[GeneratorVersion]:
        """Get all versions."""
        return list(self._versions)
    
    def num_versions(self) -> int:
        """Get number of versions."""
        return len(self._versions)
    
    def is_empty(self) -> bool:
        """Check if registry is empty."""
        return len(self._versions) == 0
    
    def sample(
        self, 
        policy: str = "uniform",
        latest_bias: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> GeneratorVersion:
        """
        Sample a generator version according to policy.
        
        Args:
            policy: "uniform", "latest_biased", or "latest"
            latest_bias: probability of sampling latest when using latest_biased
            rng: random number generator
            
        Returns:
            Sampled GeneratorVersion
        """
        if not self._versions:
            raise RuntimeError("No versions in registry")
        
        rng = rng or random.Random()
        
        if len(self._versions) == 1 or policy == "latest":
            return self._versions[-1]
        
        if policy == "uniform":
            return rng.choice(self._versions)
        
        if policy == "latest_biased":
            if rng.random() < latest_bias:
                return self._versions[-1]
            else:
                # Sample from all except latest
                return rng.choice(self._versions[:-1]) if len(self._versions) > 1 else self._versions[0]
        
        # Default: uniform
        return rng.choice(self._versions)
    
    def sample_batch(
        self,
        batch_size: int,
        policy: str = "uniform",
        latest_bias: float = 0.7,
        rng: Optional[random.Random] = None,
    ) -> List[GeneratorVersion]:
        """Sample a batch of versions for training."""
        return [self.sample(policy, latest_bias, rng) for _ in range(batch_size)]
    
    def save(self) -> None:
        """Save registry to disk."""
        os.makedirs(self.save_dir, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "versions": [v.to_dict() for v in self._versions],
        }
        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self) -> None:
        """Load registry from disk."""
        if not os.path.exists(self._registry_path):
            return
        
        with open(self._registry_path, "r") as f:
            data = json.load(f)
        
        self._next_id = data.get("next_id", 1)
        self._versions = [
            GeneratorVersion.from_dict(v) for v in data.get("versions", [])
        ]
    
    def summary(self) -> str:
        """Get a summary string of the registry."""
        lines = [f"GeneratorRegistry ({len(self._versions)} versions):"]
        for v in self._versions:
            metrics_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in v.metrics.items())
            lines.append(f"  v{v.version_id}: {os.path.basename(v.checkpoint_path)} | {metrics_str}")
        return "\n".join(lines)
