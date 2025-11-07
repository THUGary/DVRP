# Adversarial DVRP Framework Architecture

This repo provides an adversarial training setup where a diffusion-based demand generator produces problem instances and a planner solves them. The generator is trained to minimize the planner's performance.

Key modules:
- environment/: GridEnvironment defines the MDP and reward, with first-wins collision resolution.
- agent/generator/: Demand generators (rule/net). `Demand` dataclass defines a demand item.
- agent/planner/: Planners (greedy, model-based, etc.). `AgentState` dataclass and planner API.
- agent/controller/: Controllers convert targets into per-step moves.
- models/: Diffusion model for demand generation; planner model components.
- adversarial/: New unified API for building env/planner/diffusion and training.
- scripts/: Entrypoints for running episodes and training.

Data flow per step:
1) Env.step collects new demands from generator (if attached), removes expired, and applies actions.
2) Collisions: if multiple agents move to the same non-depot cell, the lowest-index agent moves; others revert (first-wins).
3) Serving: agent on a demand cell with sufficient capacity serves it fully.
4) Reward: 10*served_capacity - distance_traveled + expiry_penalty + small switch penalty.

Adversarial training loop:
- Sample demands with diffusion given condition vector.
- Rollout env with a fixed planner and controller.
- Use REINFORCE-style weighting on diffusion noise-prediction loss with EMA baseline.

See docs/api.md for interface contracts.
