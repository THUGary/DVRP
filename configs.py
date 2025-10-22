from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass
class Config:
	# Environment
	width: int = 20
	height: int = 20
	num_agents: int = 5
	capacity: int = 10
	depot: Tuple[int, int] = (0, 0)
	max_time: int = 50

	# Generator params
	generator_type: str = "rule"  # "rule" | "net"
	generator_params: Dict[str, Any] = field(default_factory=lambda: {"max_per_step": 2, "max_c": 1})

	# Planner params
	planner_type: str = "rule"  # "rule" | "net"
	planner_params: Dict[str, Any] = field(default_factory=dict)

	# Controller params
	controller_type: str = "rule"
	controller_params: Dict[str, Any] = field(default_factory=dict)


def get_default_config() -> Config:
	return Config()
