# adversarial

This package contains code and utilities for adversarial training of the
diffusion-based demand generator against fixed planners. The goal is to
train a generator that produces demand instances which are challenging for
the chosen planner (minimizing planner reward / maximizing generator reward).

Contents
- `builders.py`: helper functions to build environment, planner and diffusion model instances used by adversarial experiments.
- `types.py`: lightweight dataclasses and Protocol types used across the adversarial code (EpisodeResult, GeneratorPolicy, PlannerPolicy, ...).
- `trainers.py`: training loop(s) and helper routines. Contains `DiffusionAdversarialTrainer` which implements an adversarial REINFORCE-style update on the diffusion generator.
- `train_adversarial.py`: CLI entrypoint that wires builders and trainers into a runnable experiment script.
- `co_train/`: scaffolding for iterative co-training (coevolution) between planner model and diffusion generator.

Quick usage

From the repository root you can run the CLI script:

```bash
python adversarial/train_adversarial.py --episodes 100 --planner greedy --device cpu
```

Or import the package programmatically:

```python
from adversarial import build_env, build_planner, build_diffusion, DiffusionAdversarialTrainer, AdvConfig

env, cfg = build_env()
planner = build_planner('greedy', cfg, device='cpu')
model, condition = build_diffusion(cfg, device='cpu')
adv_cfg = AdvConfig(randomize_depot=True, lr=1e-4)
trainer = DiffusionAdversarialTrainer(env, model, condition, cfg, device='cpu', adv_cfg=adv_cfg)
```

Notes and maintenance
- The package re-exports a curated public API in `adversarial.__init__` so
  users can import high-level symbols directly from `adversarial`.
- To regenerate the project-wide API summary run: `python scripts/generate_api.py`.
- Consider adding further trainers or utilities in this package; follow the
  existing style (builders + trainer + CLI) for consistency.
- Co-training: Use `adversarial.co_train.coevolution_loop` to alternate planner and generator updates. Example:

```python
from adversarial.co_train import coevolution_loop, CoevolutionConfig

cfg = CoevolutionConfig(num_cycles=3, planner_epochs_per_cycle=5, generator_epochs_per_cycle=5)
coevolution_loop(cfg, planner_type="model")
```

If you'd like, I can also:
- add unit tests for `trainers.rollout_episode` and `builders.build_diffusion`,
- integrate a small example script under `examples/` that runs a single adversarial episode.
