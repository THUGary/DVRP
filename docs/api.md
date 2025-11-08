# Public API Contracts

## Environment
- class `GridEnvironment(width, height, num_agents, capacity, depot, generator=None, max_time=100, ...)`
- Methods:
  - `reset(seed: Optional[int]) -> ObservationDict`
  - `step(actions: List[Action]) -> (ObservationDict, float, bool, info)`
- Collision: first-wins (lowest agent index keeps the contested non-depot cell); others revert to previous positions.
- ObservationDict keys:
  - `time: int`
  - `depot: Tuple[int,int]`
  - `agent_states: List[Tuple[int,int,int]]` # (x, y, remaining_capacity)
  - `demands: List[Tuple[int,int,int,int,int]]` # (x, y, t_arrival, demand, t_due), filtered by t <= time

## Generator
- Dataclass `Demand(x:int, y:int, t:int, c:int, end_t:int)`
- Abstract `BaseDemandGenerator`:
  - `reset(seed)`
  - `sample(t) -> List[Demand]`
- Diffusion model `DemandDiffusionModel`:
  - `sample(condition, num_demands, grid_size) -> Tensor[N,5]` with normalized features `(t, x, y, c, lifetime)`
  - `__call__(x_start, condition) -> (noise, predicted_noise)` for noise-prediction loss

## Planner
- Dataclass `AgentState(x:int, y:int, s:int)`
- Abstract `BasePlanner.plan(observations, agent_states, depot, t, horizon=1, ...) -> List[Deque[Target]]`
- Targets are `(x,y)` grid coordinates.

## Controller
- Abstract `BaseController.act(current_pos: (x,y), target_queue: Deque[Target]) -> Action(dx,dy)`
- Default `RuleBasedController`: one-step toward queue head; if at target, pop and stay.

## Adversarial API (new)
- `adversarial.builders.build_env(cfg=None) -> (GridEnvironment, Config)`
- `adversarial.builders.build_planner(planner_type, cfg, device, ckpt=None)`
- `adversarial.builders.build_diffusion(cfg, device, init_ckpt=None, num_steps=1000) -> (model, condition)`
- `adversarial.trainers.DiffusionAdversarialTrainer(env, model, condition, cfg, device, adv_cfg)`
  - `.train(planner, episodes, renderer=None, save_path=None, seed=1)`

## Types
- Action = `(dx:int, dy:int)` with each in `{-1,0,1}`
- Demand tuple = `(x:int, y:int, t:int, c:int, end_t:int)`
- Reward shaping and penalties are described in docs/architecture.md
