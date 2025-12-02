from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .rule_generator import RuleBasedGenerator
from .static_wrappers import StaticDemandGenerator
from .base import BaseDemandGenerator


def _clone_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return dict(params) if params else {}


def build_rule_based_generator(
    width: int,
    height: int,
    generator_params: Optional[Dict[str, Any]] = None,
    *,
    depot: Optional[Tuple[int, int]] = None,
    static_demands: bool = False,
    max_end_time: Optional[int] = None,
) -> BaseDemandGenerator:
    """Create a rule-based generator with consistent parameter sanitization."""

    params = _clone_params(generator_params)
    # Ensure depot is injected only once.
    if depot is not None:
        params["depot"] = depot
    base_gen = RuleBasedGenerator(width, height, **params)
    if static_demands:
        resolved_end = None if max_end_time is None else int(max_end_time)
        return StaticDemandGenerator(base_gen, max_end_time=resolved_end)
    return base_gen


__all__ = ["build_rule_based_generator"]
