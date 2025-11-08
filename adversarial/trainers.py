"""Deprecated stub: diffusion adversarial trainer moved to training/generator.

This module re-exports symbols from training.generator.adversarial_trainer to
keep backward compatibility while concentrating training code under
`training/generator` as requested.
"""

from training.generator.adversarial_trainer import (
    DiffusionAdversarialTrainer,
    AdvConfig,
    rollout_episode,
    _generate_demands,
)

__all__ = [
    "DiffusionAdversarialTrainer",
    "AdvConfig",
    "rollout_episode",
    "_generate_demands",
]
