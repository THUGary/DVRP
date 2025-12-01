"""
Adversarial V2 Training Package

Co-evolution training framework for V2Planner and Diffusion Generator.

Key components:
- CoevolutionConfig, EnvironmentConfig: Configuration dataclasses (defined in config.py)
- GeneratorRegistry: Track all generator versions to avoid policy cycling  
- DemandConverter: Convert diffusion output to static/dynamic demands
- PlannerTrainer: Train V2Planner using multi-version generator distributions
- GeneratorTrainer: Adversarial training to find planner weaknesses
- coevolution_loop: Main alternating training loop

All CLI argument processing is done in cotrain.py (single entry point).
Config objects are then passed to other modules.

Model hyperparameters use defaults from:
- Planner: training_v2/train_static.py
- Generator: training/generator/adversarial_trainer.py (AdvConfig)

Usage (command line):
    python -m adversarial_v2 --mode static --num-cycles 10
    
Usage (Python):
    from adversarial_v2 import coevolution_loop, CoevolutionConfig, EnvironmentConfig
    
    env_config = EnvironmentConfig(map_size=20)
    config = CoevolutionConfig(mode="static", num_cycles=10, env=env_config)
    coevolution_loop(config)
"""

from .config import CoevolutionConfig, EnvironmentConfig
from .utils.registry import GeneratorRegistry, GeneratorVersion
from .utils.demand_converter import DemandConverter, DemandTuple
from .train_planner import PlannerTrainer
from .train_generator import GeneratorTrainer
from .coevolution import coevolution_loop

__all__ = [
    # Configuration dataclasses
    "CoevolutionConfig",
    "EnvironmentConfig",
    # Registry
    "GeneratorRegistry",
    "GeneratorVersion",
    # Converters
    "DemandConverter",
    "DemandTuple",
    # Trainers
    "PlannerTrainer",
    "GeneratorTrainer",
    # Main function
    "coevolution_loop",
]