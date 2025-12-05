from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .rule_generator import RuleBasedGenerator
from .static_wrappers import StaticDemandGenerator
from .base import BaseDemandGenerator, Demand


def _clone_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return dict(params) if params else {}


class PregeneratedDemandGenerator(BaseDemandGenerator):
    """
    A generator that returns pre-generated demands.
    
    Used for evaluating planners on diffusion-generated distributions
    without having to modify the environment significantly.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        demands: List[Tuple[int, int, int, int, int]],
        max_end_time: Optional[int] = None,
    ):
        """
        Args:
            width: Grid width
            height: Grid height
            demands: List of (x, y, t, c, end_t) tuples
            max_end_time: Override end_t for static mode
        """
        super().__init__(width, height)
        self._demands = demands
        self._max_end_time = max_end_time
        self._generated = False
    
    def reset(self, seed: Optional[int] = None):
        self._generated = False
    
    def sample(self, t: int) -> List[Demand]:
        """
        Sample demands at the current time.
        
        For static mode (t=0 for all), returns all demands on first call.
        For dynamic mode, returns demands whose t <= current_time.
        """
        if not self._generated:
            self._generated = True
            # Return all demands with adjusted end_t if max_end_time specified
            if self._max_end_time is not None:
                demands = [(x, y, t, c, self._max_end_time) for x, y, t, c, end_t in self._demands]
            else:
                demands = list(self._demands)
            
            # Convert to Demand objects
            return [Demand(x=x, y=y, t=dt, c=c, end_t=end_t) for x, y, dt, c, end_t in demands]
        return []


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
    
    # Check for pre-generated demands (from diffusion model evaluation)
    pregenerated = params.pop("_pregenerated_demands", None)
    if pregenerated is not None:
        # Use pre-generated demands instead of rule-based generation
        resolved_end = max_end_time if static_demands else None
        return PregeneratedDemandGenerator(
            width, height, 
            demands=pregenerated,
            max_end_time=resolved_end,
        )
    
    # Ensure depot is injected only once.
    if depot is not None:
        params["depot"] = depot
    base_gen = RuleBasedGenerator(width, height, **params)
    if static_demands:
        resolved_end = None if max_end_time is None else int(max_end_time)
        return StaticDemandGenerator(base_gen, max_end_time=resolved_end)
    return base_gen


__all__ = ["build_rule_based_generator", "PregeneratedDemandGenerator"]
