## DVRP-10.28 Overview

Dynamic Vehicle Routing on a grid with modular components for demand generation, planning, and control. Vehicles operate in discrete time, obey capacity limits, and may share the depot cell while avoiding collisions elsewhere.

### Key Features
- Grid environment (`environment/env.py`) that enforces demand expiry, capacity refills at the depot, and anti-collision logic.
- Demand generators (`agent/generator/`) including the rule-based baseline and a diffusion-model interface.
- Planners (`agent/planner/`) spanning greedy heuristics, multi-heuristic variants (`fri`, `rbso`, `dcp`), and a neural DVRPNet-based `ModelPlanner`.
- Controllers (`agent/controller/`) that translate target queues into single-step moves.
- Live visualization via `utils/pygame_renderer.py` when `--render` is used.

---

## Installation

```bash
python -m venv .venv            # optional virtualenv
source .venv/bin/activate       # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch/CUDA is optional but required to run planners or generators that rely on neural checkpoints.

---

## Running Simulations (`run.py`)

Rule-based baseline:

```bash
python run.py --seed 0
```

Render the episode (opens a Pygame window):

```bash
python run.py --seed 0 --render --fps 10
```

Use the neural planner with a checkpoint (agents can share the depot only):

```bash
python run.py --model --ckpt checkpoints/planner/planner_20_2_200.pt --render
```

Key CLI arguments:
- `--seed`: reproducibility for depot randomization and RNG-dependent components.
- `--planner`: `greedy`, `fri`, `rbso`, `dcp`, or `model`.
- `--model`: convenience flag equivalent to `--planner model`.
- `--render`, `--fps`: visualization control.
- Model planner extras: `--ckpt`, `--time-plan`, `--planner-device`, `--lateness-lambda`, `--d-model`, `--nhead`, `--nlayers`.

Each episode randomizes the depot position (seeded) and updates generator/depot references accordingly. Agents start stacked on the depot with full capacity; non-depot collisions are automatically prevented.

---

## Project Structure

```
configs.py              # default configuration (width, height, generator params, etc.)
run.py                  # main entry point tying env, planner, controller together
test.py                 # smoke tests for the rule-based stack
test_model.py           # legacy DVRPNet evaluation script (optional)
train_model.py          # training loop for the DVRPNet planner
agent/
  generator/            # rule-based + diffusion demand generators
  planner/              # rule-based, heuristic, and neural planners (ModelPlanner, etc.)
  controller/           # RuleBasedController and helpers
environment/
  env.py                # GridEnvironment implementation with collision handling
models/
  generator_model/      # diffusion model for demand generation
  planner_model/        # DVRPNet (encoder/decoder/layers)
scripts/
  generate_data.py      # offline dataset generation (rule-based)
  normalize_data.py     # dataset normalization utilities
  train_diffusion_generator.py
utils/
  pygame_renderer.py    # visualization
  state_manager.py      # planning state tracking helpers
checkpoints/            # pretrained diffusion planner/generator weights
runs/                   # TensorBoard logs from diffusion training
```

`run_all.sh` provides an example batch script for automated experiments.

---

## Diffusion Demand Generator Workflow

1. Configure sampling ranges in `configs.py` (`GENERATOR_PARAM_SPACE`).
2. Generate raw data:
   ```bash
   python scripts/generate_data.py
   ```
3. Normalize and split datasets:
   ```bash
   python scripts/normalize_data.py
   ```
4. Train the diffusion generator:
   ```bash
   python scripts/train_diffusion_generator.py
   ```
5. Set `generator_type = "net"` in `configs.py` (or override via CLI) and provide the learned checkpoint under `checkpoints/diffusion_model.pth`.

High batch sizes (e.g., 4096) stabilize diffusion training but require ample GPU memory (~30 GB). Adjust according to hardware and monitor TensorBoard logs in `runs/`.

---

## Planner Model Training

`train_model.py` trains the DVRPNet planner on datasets prepared by `data_gen.py`. Checkpoints are saved to `checkpoints/planner/` (e.g., `planner_20_2_200.pt`) and can be invoked from `run.py --planner model` or `run.py --model`.

