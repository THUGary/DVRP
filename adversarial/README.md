# adversarial

This package contains code and utilities for adversarial training of the
diffusion-based demand generator against fixed planners. The goal is to
train a generator that produces demand instances which are challenging for
the chosen planner (minimizing planner reward / maximizing generator reward).

Contents (after cleanup: only co-training logic retained here)
- `builders.py`: helper functions to build environment, planner and diffusion model instances.
- `types.py`: lightweight dataclasses / Protocol types.
- `co_train/`: scaffolding for iterative co-training (coevolution) between planner model and diffusion generator (`train_coevolution.py`, registry, scheduler).

Quick usage (coevolution only)

Example: run co-training loop directly in Python

```python
from adversarial import coevolution_loop, CoevolutionConfig, build_env, build_planner, build_diffusion
cfg = CoevolutionConfig(num_cycles=3, planner_epochs_per_cycle=5, generator_epochs_per_cycle=5, sample_latest_prob=0.8)
coevolution_loop(cfg, planner_type="model")
```

Notes and maintenance
- Public API now focuses on co-training (`coevolution_loop`, `CoevolutionConfig`). Diffusion-only adversarial training scripts were removed from this folder to reduce clutter; the generator training logic itself lives under `training/generator/`.
- You can still build base components via `build_env`, `build_planner`, `build_diffusion` for custom hooks.
- Future work ideas:
  - Add CLI wrapper for coevolution (e.g., `python -m adversarial.co_train.train_coevolution` if a main is added).
  - Add unit tests for registry & scheduler sampling.
  - Provide evaluation script comparing planner performance across generator versions.
