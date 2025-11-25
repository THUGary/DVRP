from __future__ import annotations
import argparse
import random
import os
from typing import List, Tuple
from dataclasses import replace
import copy


class StaticDemandGenerator:
	"""Wrap any BaseDemandGenerator so every demand appears at t=0 (static scenario)."""

	def __init__(self, base_generator, *, full_window_end_t: int | None = None):
		self._base = base_generator
		self.width = base_generator.width
		self.height = base_generator.height
		self.params = getattr(base_generator, "params", {})
		self.max_time = getattr(base_generator, "max_time", self.params.get("max_time", 1))
		self._snapshot = []
		self._released = False
		self._full_window_end_t = full_window_end_t

	def reset(self, seed: int | None = None):
		self._base.reset(seed)
		self._snapshot.clear()
		self._released = False
		max_time = int(getattr(self._base, "max_time", self.params.get("max_time", 1)))
		max_time = max(1, max_time)
		full_window_end_t = self._full_window_end_t if self._full_window_end_t is not None else max_time
		for t in range(max_time):
			for demand in self._base.sample(t):
				static_demand = replace(demand, t=0, end_t=full_window_end_t)
				self._snapshot.append(static_demand)

	def sample(self, t: int):
		if t == 0 and not self._released:
			self._released = True
			return list(self._snapshot)
		return []

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.controller import RuleBasedController
from utils.pygame_renderer import PygameRenderer
from utils.state_manager import PlanningState, update_planning_state
from agent.generator.base import BaseDemandGenerator
from agent.planner.base import BasePlanner
from agent.planner import RuleBasedPlanner
from agent.planner import FastReactiveInserter
from agent.planner import RepairBasedStabilityOptimizer
from agent.planner import DistributedCooperativePlanner
from agent.planner import ModelPlanner
from agent.planner import CVRPPOMOPlanner



def build_env(cfg: Config, planner_type: str, *, static_demands: bool = False) -> Tuple[GridEnvironment, BaseDemandGenerator, BasePlanner, RuleBasedController]:
	# choose generator class by config
	if cfg.generator_type == "net":
		# lazy import to avoid unnecessary dependencies when not used
		from agent.generator.net_generator import NetDemandGenerator as GenClass
	else:
		from agent.generator import RuleBasedGenerator as GenClass

	gen = GenClass(cfg.width, cfg.height, **cfg.generator_params)
	max_end_time_cfg = getattr(cfg, "max_end_time", None)
	max_end_time = int(max_end_time_cfg if max_end_time_cfg is not None else cfg.max_time * 2)
	if static_demands:
		gen = StaticDemandGenerator(gen, full_window_end_t=max_end_time)
	env = GridEnvironment(
		width=cfg.width,
		height=cfg.height,
		num_agents=cfg.num_agents,
		capacity=cfg.capacity,
		depot=cfg.depot,
		generator=gen,
		max_time=cfg.max_time,
		expiry_penalty_scale=float(getattr(cfg, "expiry_penalty_scale", 5.0)),
		switch_penalty_scale=float(getattr(cfg, "switch_penalty_scale", 0.01)),
		capacity_reward_scale=float(getattr(cfg, "capacity_reward_scale", 10.0)),
		exploration_history_n=int(getattr(cfg, "exploration_history_n", 0)),
		exploration_penalty_scale=float(getattr(cfg, "exploration_penalty_scale", 0.0)),
		wait_penalty_scale=float(getattr(cfg, "wait_penalty_scale", 0.001)),
		max_end_time=max_end_time,
		include_service_time=bool(getattr(cfg, "include_service_time", False)),
	)
	env.num_agents = cfg.num_agents
	if planner_type == "greedy":
		# 使用 Rule-based Planner（需要显式传入 full_capacity 来自 Config.capacity）
		planner = RuleBasedPlanner(full_capacity=cfg.capacity, **cfg.planner_params)
	elif planner_type == "fri":
		# 使用 Fast Reactive Inserter
		planner = FastReactiveInserter()
	elif planner_type == "rbso":
		# 使用 Repair-based Stability Optimizer（带参数）
		planner = RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
	elif planner_type == "dcp":
		# 使用 Distributed Cooperative Planner（带参数）
		planner = DistributedCooperativePlanner(auction_rounds=5, bid_strategy='time_urgency')
	elif planner_type == "model":
		planner_params = dict(cfg.model_planner_params)
		ckpt_path = planner_params.pop("ckpt", None)
		planner = ModelPlanner(full_capacity=cfg.capacity, **planner_params)
		if ckpt_path:
			if hasattr(planner, "load_from_ckpt"):
				planner.load_from_ckpt(ckpt_path)
			else:
				raise RuntimeError("Selected planner does not support checkpoint loading.")
	elif planner_type == "cvrp_pomo":
		cvrp_params = dict(cfg.cvrp_planner_params)
		cvrp_params.pop("enabled", None)
		pomo_root = cvrp_params.pop("pomo_root", None)
		if not pomo_root:
			raise ValueError("Config.cvrp_planner_params must define 'pomo_root' when using the CVRP planner.")
		env_params = copy.deepcopy(cvrp_params.pop("env_params", {}))
		model_params = copy.deepcopy(cvrp_params.pop("model_params", {}))
		checkpoint = cvrp_params.pop("checkpoint", None)
		device_override = cvrp_params.pop("device", "cpu")
		max_nodes = cvrp_params.pop("max_nodes", env_params.get("problem_size", cfg.capacity))
		coord_norm = cvrp_params.pop("coord_normalizer", None)
		selection_policy = cvrp_params.pop("selection_policy", "earliest_due")
		planner = CVRPPOMOPlanner(
			pomo_root=pomo_root,
			env_params=env_params,
			model_params=model_params,
			checkpoint=checkpoint,
			device=device_override,
			max_nodes=max_nodes,
			coord_normalizer=coord_norm,
			grid_width=cfg.width,
			grid_height=cfg.height,
			capacity=cfg.capacity,
			selection_policy=selection_policy,
			**cvrp_params,
		)
	else:
		raise ValueError(f"Unknown planner type: {planner_type}")
	controller = RuleBasedController(**cfg.controller_params)
	return env, gen, planner, controller


def run_episode(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, planner: str = "greedy", *, static_demands: bool = False) -> None:
	# deterministically randomize depot location per episode
	rng = random.Random(seed)
	depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))
	cfg.depot = depot
	cfg.generator_params = {**cfg.generator_params, "depot": depot}
	#print model used
	print(f"Using planner: {planner}")
	planner_type = planner
	if planner_type == "model":
		ckpt_info = cfg.model_planner_params.get("ckpt")
		if ckpt_info:
			print(f"Loading model checkpoint: {ckpt_info}")
	elif planner_type == "cvrp_pomo":
		params = cfg.cvrp_planner_params
		print(f"CVRP-POMO root: {params.get('pomo_root')} | ckpt: {params.get('checkpoint')}")
	env, gen, planner_impl, controller = build_env(cfg, planner_type=planner_type, static_demands=static_demands)
	obs = env.reset(seed)
	total_reward = 0.0
	done = False
	step = 0
	renderer = None

	# 初始化规划状态管理器
	planning_state = PlanningState()
	planning_state.reset(cfg.num_agents)

	# 记录上一步的需求，用于检测新增需求
	prev_demands = []
	total_demand = 0

	if render:
		renderer = PygameRenderer(cfg.width, cfg.height)
		renderer.init()

	while not done:
		# 检测新增的需求
		current_demands = obs["demands"]
		new_demands = [d for d in current_demands if d not in prev_demands]
		total_demand += len(new_demands)

		# 更新规划状态（在规划之前）
		agent_states = obs["agent_states"]  # list of (x,y,s)
		update_planning_state(
			planning_state=planning_state,
			agent_states=agent_states,
			new_demands=new_demands,
			obs_demands=current_demands,
		)

		# Plan targets using current observation with enhanced information
		agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
		plan_horizon = 1
		if planner_type == "model":
			cfg.model_planner_params["time_plan"] = 1
		targets = planner_impl.plan(
			observations=obs["demands"],  # [(x, y, t_arrival, c, t_due), ...]
			agent_states=agents,
			depot=obs["depot"],
			t=obs["time"],
			horizon=plan_horizon,
			current_plans=planning_state.current_plans,  # 新增：当前规划路径
			global_nodes=planning_state.global_nodes.nodes,  # 新增：全局节点列表 [(x, y, t_arrival, t_due, demand), ...]
			serve_mark=planning_state.global_nodes.serve_mark,  # 新增：服务标记
			unserved_count=planning_state.get_unserved_count(),  # 新增：未服务节点数量
		)
		print(f"[PLANNER] step={step} selections={{}}".format([list(target) for target in targets]))

		# 更新规划结果到状态管理器
		planning_state.update_plans(targets)

		# Controller decides per-agent move
		actions: List[Tuple[int, int]] = []
		for i, (x, y, s) in enumerate(agent_states):
			actions.append(controller.act((x, y), targets[i]))

		# 执行动作并更新环境
		obs, reward, done, info = env.step(actions)
		prev_demands = list(current_demands)

		if renderer is not None:
			if not renderer.render(obs):
				break
			# throttle
			if fps > 0:
				import time
				time.sleep(1.0 / fps)
		total_reward += reward
		step += 1
		if step % 10 == 0 or done:
			unserved = planning_state.get_unserved_count()
			print(f"Step {step:03d} | time={obs['time']} | reward={reward:.0f} | total={total_reward:.0f} | demands={len(obs['demands'])} | total_demand={total_demand} | unserved={unserved}")
	print(f"Episode done in {step} steps. Total reward={total_reward:.0f}. Total demand encountered={total_demand}")
	if renderer is not None:
		renderer.close()


def main() -> None:
	# --------------------------------------------------------------
	# Minimal CLI: keep only seed, render, fps, pmodel, gmodel
	# --------------------------------------------------------------
	parser = argparse.ArgumentParser(description="DVRP runner")
	parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
	parser.add_argument("--render", action="store_true", help="Use pygame to visualize")
	parser.add_argument("--fps", type=int, default=10, help="Render FPS when --render (default: 10)")
	# --pmodel optionally accepts a checkpoint path; if omitted, use default from cfg
	parser.add_argument("--pmodel", nargs="?", const="__DEFAULT__", help="Use model planner; optionally pass checkpoint path (.pt/.pth). Example: --pmodel checkpoints/planner/planner_rl_best.pt")
	# --planner explicitly chooses planner type and overrides --pmodel/--cvrp when provided
	parser.add_argument("--planner", choices=["greedy", "model", "cvrp_pomo", "fri", "rbso", "dcp"], default=None, help="Explicitly select planner type; overrides --pmodel/--cvrp flags")
	parser.add_argument("--gmodel", action="store_true", help="Use neural net demand generator; otherwise rule")
	parser.add_argument("--service-time", action="store_true", help="Enable service times for demands (vehicles must remain on-site before completion)")
	parser.add_argument("--num-agents", type=int, default=2, help="Override number of agents for the episode (overrides config)")
	parser.add_argument("--map-wid", type=int, default=None, help="Override map width")
	parser.add_argument("--map-hei", type=int, default=None, help="Override map height")
	parser.add_argument("--total-demand", type=int, default=None, help="Override total demand parameter for the generator")
	parser.add_argument("--cvrp", action="store_true", help="Use the CVRP-POMO planner adapter instead of DVRP planners")
	parser.add_argument("--cvrp-root", type=str, default=None, help="Override path to the CVRP/POMO folder (defaults to config)")
	parser.add_argument("--cvrp-ckpt", type=str, default=None, help="Override path to the CVRP checkpoint (.pt)")
	parser.add_argument("--static-demands", action="store_true", help="Release all demands at time 0 to visualize static VRP instances")
	args = parser.parse_args()

	cfg = get_default_config()
	cfg.include_service_time = bool(args.service_time)
	# override number of agents if provided on CLI
	if args.num_agents is not None and args.num_agents > 0:
		cfg.num_agents = int(args.num_agents)
	if args.map_wid is not None and args.map_wid > 0:
		cfg.width = int(args.map_wid)
	if args.map_hei is not None and args.map_hei > 0:
		cfg.height = int(args.map_hei)
	if args.total_demand is not None and args.total_demand > 0:
		cfg.generator_params["total_demand"] = int(args.total_demand)

	# Planner selection precedence:
	# 1) --planner if provided (explicit)
	# 2) --cvrp flag
	# 3) --pmodel presence
	planner_choice = None
	use_model_planner = False
	if args.planner is not None:
		planner_choice = args.planner
		if planner_choice == "model":
			use_model_planner = True
			cfg.planner_type = "model"
		elif planner_choice == "cvrp_pomo":
			cfg.planner_type = "cvrp_pomo"
		else:
			cfg.planner_type = planner_choice
	else:
		# fallback to existing flags
		use_model_planner = args.pmodel is not None
		if args.cvrp and use_model_planner:
			raise ValueError("--cvrp and --pmodel are mutually exclusive")
		if args.cvrp:
			planner_choice = "cvrp_pomo"
			cfg.planner_type = "cvrp_pomo"
			if args.cvrp_root:
				cfg.cvrp_planner_params["pomo_root"] = args.cvrp_root
			if args.cvrp_ckpt:
				cfg.cvrp_planner_params["checkpoint"] = args.cvrp_ckpt
		else:
			planner_choice = "model" if use_model_planner else "greedy"
			if use_model_planner:
				cfg.planner_type = "model"

	# If using model planner and a path was provided via --pmodel, resolve it into cfg
	if use_model_planner and isinstance(args.pmodel, str) and args.pmodel != "__DEFAULT__":
		raw = os.path.expanduser(args.pmodel)
		candidates = [
			raw,
			os.path.join("checkpoints", raw) if not os.path.isabs(raw) else raw,
			os.path.join("checkpoints", "planner", os.path.basename(raw)),
		]
		ckpt_path = None
		for p in candidates:
			p_abs = os.path.abspath(p)
			if os.path.isfile(p_abs):
				ckpt_path = p_abs
				break
		# If not found, still set the expanded absolute path for downstream attempt
		if ckpt_path is None:
			ckpt_path = os.path.abspath(raw)
			print(f"WARNING: Planner checkpoint not found at {ckpt_path}. Will attempt to load; ensure path is correct.")
		cfg.model_planner_params["ckpt"] = ckpt_path

	# Generator: net if --gmodel else keep default (rule)
	if args.gmodel:
		cfg.generator_type = "net"

	# Run
	run_episode(cfg, seed=args.seed, render=args.render, fps=args.fps, planner=planner_choice, static_demands=args.static_demands)


if __name__ == "__main__":
	main()