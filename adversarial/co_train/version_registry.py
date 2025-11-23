from __future__ import annotations
"""Generator version registry.

Stores historical generator checkpoints (paths + metadata) so that
planner training can sample from all previous generator distributions.

Note: This is a lightweight in-memory + filesystem linked registry.
Persisting as JSON/YAML could be added later.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time, os, torch


@dataclass
class GeneratorVersion:
    version_id: int
    ckpt_path: str
    timestamp: float = field(default_factory=lambda: time.time())
    metrics: Dict[str, Any] = field(default_factory=dict)

    def load(self, device: str | torch.device = "cpu"):
        """Load the generator model state dict. Caller constructs model object."""
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(self.ckpt_path)
        return torch.load(self.ckpt_path, map_location=device)


class GeneratorVersionRegistry:
    def __init__(self) -> None:
        self._versions: List[GeneratorVersion] = []
        self._next_id = 1

    def add(self, ckpt_path: str, metrics: Optional[Dict[str, Any]] = None) -> GeneratorVersion:
        gv = GeneratorVersion(version_id=self._next_id, ckpt_path=ckpt_path, metrics=metrics or {})
        self._versions.append(gv)
        self._next_id += 1
        return gv

    def list(self) -> List[GeneratorVersion]:
        return list(self._versions)

    def latest(self) -> Optional[GeneratorVersion]:
        return self._versions[-1] if self._versions else None

    def is_empty(self) -> bool:
        return len(self._versions) == 0

    def summary(self) -> str:
        lines = ["GeneratorVersionRegistry:"]
        for v in self._versions:
            lines.append(f"  - id={v.version_id} path={v.ckpt_path} metrics={v.metrics}")
        return "\n".join(lines)
