from __future__ import annotations
import argparse
import random
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
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
from agent.planner import V2Planner, create_v2_planner
# reuse tracker & helpers from run_evaluate for run-mode outputs
from run_evaluate import EvaluationTracker, convert_all, create_output_run_dir



def build_env(
	cfg: Config,
	planner_type: str,
	*,
	static_demands: bool = False,
	planner_kwargs: Dict[str, Any] | None = None,
) -> Tuple[GridEnvironment, BaseDemandGenerator, BasePlanner, RuleBasedController]:
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
		from agent.generator.static_rule_gen import StaticDemandGen
		gen = StaticDemandGen(cfg.width, cfg.height, **cfg.generator_params)
		# gen = StaticDemandGenerator(gen, full_window_end_t=max_end_time)
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
		depot_return_bonus_scale=float(getattr(cfg, "depot_return_bonus_scale", 0.0)),
		max_end_time=max_end_time,
		include_service_time=bool(getattr(cfg, "include_service_time", False)),
	)
	env.num_agents = cfg.num_agents
	planner_kwargs = planner_kwargs or {}
	if planner_type in ("greedy", "rule", "optimize"):
		mode = planner_kwargs.get("mode")
		if mode is None:
			mode = "optimize" if planner_type == "optimize" else "greedy"
		planner_params = dict(cfg.planner_params)
		planner_params.pop("mode", None)
		planner = RuleBasedPlanner(full_capacity=cfg.capacity, mode=mode, **planner_params)
	elif planner_type == "fri":
		# 使用 Fast Reactive Inserter
		planner = FastReactiveInserter()
	elif planner_type == "rbso":
		# 使用 Repair-based Stability Optimizer（带参数）
		planner = RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
	elif planner_type == "dcp":
		# 使用 Distributed Cooperative Planner（带参数）
		planner = DistributedCooperativePlanner(auction_rounds=5, bid_strategy='time_urgency')
	elif planner_type in ("model", "dynamic"):
		# Use V2Planner with dynamic mode (static model + adapter)
		v2_params = dict(cfg.v2_planner_params) if hasattr(cfg, 'v2_planner_params') else {}
		static_ckpt = v2_params.pop("static_ckpt", "checkpoints/static_vrp_v2/best_n20.pt")
		adapter_ckpt = v2_params.pop("adapter_ckpt", "checkpoints/dynamic_adapter_v2/best_adapter.pt")
		device = v2_params.pop("device", "cuda")
		planner = create_v2_planner(
			mode="dynamic",
			static_checkpoint=static_ckpt,
			adapter_checkpoint=adapter_ckpt,
			device=device,
			grid_width=cfg.width,
			grid_height=cfg.height,
			full_capacity=cfg.capacity,
			max_time=cfg.max_time,
			**v2_params,
		)
	elif planner_type == "static":
		# Use V2Planner with static mode (POMO static VRP model only)
		v2_params = dict(cfg.v2_planner_params) if hasattr(cfg, 'v2_planner_params') else {}
		static_ckpt = v2_params.pop("static_ckpt", "checkpoints/static_vrp_v2/best_n20.pt")
		device = v2_params.pop("device", "cuda")
		planner = create_v2_planner(
			mode="static",
			static_checkpoint=static_ckpt,
			device=device,
			grid_width=cfg.width,
			grid_height=cfg.height,
			full_capacity=cfg.capacity,
			max_time=cfg.max_time,
			**v2_params,
		)
	else:
		raise ValueError(f"Unknown planner type: {planner_type}")
	controller = RuleBasedController(**cfg.controller_params)
	return env, gen, planner, controller


def run_episode(
	cfg: Config,
	seed: int = 0,
	render: bool = False,
	fps: int = 10,
	planner: str = "greedy",
	*,
	static_demands: bool = False,
	planner_kwargs: Optional[Dict[str, Any]] = None,
	save_run: bool = False,
	max_steps: Optional[int] = None,
) -> None:
	# deterministically randomize depot location per episode
	rng = random.Random(seed)
	depot = (rng.randint(0, cfg.width - 1), rng.randint(0, cfg.height - 1))
	cfg.depot = depot
	cfg.generator_params = {**cfg.generator_params, "depot": depot}
	#print model used
	print(f"Using planner: {planner}")
	planner_type = planner
	if planner_type in ("model", "dynamic"):
		v2_params = cfg.v2_planner_params if hasattr(cfg, 'v2_planner_params') else {}
		print(f"V2Planner (dynamic mode): static={v2_params.get('static_ckpt', 'default')} adapter={v2_params.get('adapter_ckpt', 'default')}")
	elif planner_type == "static":
		v2_params = cfg.v2_planner_params if hasattr(cfg, 'v2_planner_params') else {}
		print(f"V2Planner (static mode): static={v2_params.get('static_ckpt', 'default')}")
	env, gen, planner_impl, controller = build_env(
		cfg,
		planner_type=planner_type,
		static_demands=static_demands,
		planner_kwargs=planner_kwargs,
	)
	obs = env.reset(seed)
	total_reward = 0.0
	done = False
	step = 0
	renderer = None

	# set up run tracker to record per-step metrics for saving/plotting
	tracker = EvaluationTracker(cfg.num_agents)
	tracker.register_new_demands(obs.get("demands", []), obs.get("time", 0))
	prev_demands = list(obs["demands"])

	# 初始化规划状态管理器
	planning_state = PlanningState()
	planning_state.reset(cfg.num_agents)

	# 记录上一步的需求，用于检测新增需求
	prev_demands = []
	total_demand = 0

	# helper: convert numpy scalars in targets to native Python types for clean printing
	def _clean_for_print(obj):
		if isinstance(obj, (list, tuple)):
			return [
				_clean_for_print(x) for x in obj
			]
		if isinstance(obj, dict):
			return {k: _clean_for_print(v) for k, v in obj.items()}
		if isinstance(obj, np.generic):
			return obj.item()
		return obj

	if render:
		renderer = PygameRenderer(cfg.width, cfg.height)
		renderer.init()

	while not done:
		# Check max steps limit
		if max_steps is not None and step >= max_steps:
			print(f"Max steps ({max_steps}) reached, terminating episode.")
			break
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
			depot=obs["depot"],  # 传入 depot 以便在清理时保留 depot 目标
		)

		# Plan targets using current observation with enhanced information
		agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
		plan_horizon = 1
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
		cleaned_targets = _clean_for_print([list(target) for target in targets])
		print(f"[PLANNER] step={step} selections={cleaned_targets}")

		# 更新规划结果到状态管理器
		# Note: update_plans stores a copy, but we need controller to modify the same deques
		# So we update first, then use current_plans for controller actions
		planning_state.update_plans(targets)

		# Controller decides per-agent move
		# IMPORTANT: Use planning_state.current_plans so that popleft() persists across steps
		# This allows the agent to stay at target position for one step (to serve demand)
		actions: List[Tuple[int, int]] = []
		for i, (x, y, s) in enumerate(agent_states):
			actions.append(controller.act((x, y), planning_state.current_plans[i]))

		# 执行动作并更新环境
		obs, reward, done, info = env.step(actions)
		current_time = obs.get("time", 0)
		# detect disappeared (served or expired) demands
		disappeared = [d for d in prev_demands if d not in obs["demands"]]
		for d in disappeared:
			x, y, t, c, end_t = d
			if int(end_t) >= int(current_time):
				tracker.record_served_by_tuple(d, served_time=current_time)
			else:
				tracker.record_expired(d)

		# record agent paths and loads
		tracker.record_path_and_distance(obs["agent_states"], current_time, env)
		prev_demands = list(obs["demands"])

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

	# Save run outputs (metrics + plots) under outputs/run/<ts>/ only when requested
	if save_run:
		try:
			metrics = tracker.finalize()
			# use `outputs/run` as base directory
			output_root = os.path.join("outputs", "run")
			run_run_dir = create_output_run_dir(output_root, "run")
			metrics_safe = convert_all(metrics)
			import json
			with open(os.path.join(run_run_dir, "metrics.json"), "w") as f:
				json.dump(metrics_safe, f, indent=2)
			with open(os.path.join(run_run_dir, "paths_output.json"), "w") as f:
				json.dump(convert_all(metrics.get("paths", [])), f, indent=2)
			print(f"Run outputs saved -> {run_run_dir}")
			# Generate simple plots
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
			# (a) loads per step
			loads = metrics.get("loads_per_step", [])
			if loads:
				max_len = max(len(l) for l in loads)
				plt.figure(figsize=(10, 6))
				for i, l in enumerate(loads):
					if len(l) < max_len and len(l) > 0:
						l_ext = list(l) + [l[-1]] * (max_len - len(l))
					else:
						l_ext = list(l)
					plt.plot(range(len(l_ext)), l_ext, label=f"agent_{i}")
				plt.xlabel("time_step")
				plt.ylabel("carried_load")
				plt.title("Agent carried load over time")
				plt.legend(loc="upper right")
				fname = os.path.join(run_run_dir, "loads_over_time.png")
				plt.tight_layout()
				plt.savefig(fname, dpi=200)
				plt.close()
				print(f"Saved plot: {fname}")

			# (b) cumulative agent distance
			paths = metrics.get("paths", [])
			if paths:
				plt.figure(figsize=(10, 6))
				for i, p in enumerate(paths):
					if not p:
						continue
					times = [int(t[0]) for t in p]
					xs = [int(t[1]) for t in p]
					ys = [int(t[2]) for t in p]
					cum = []
					acc = 0
					lastx, lasty = xs[0], ys[0]
					cum.append(acc)
					for x, y in zip(xs[1:], ys[1:]):
						d = abs(int(x) - int(lastx)) + abs(int(y) - int(lasty))
						acc += d
						cum.append(acc)
						lastx, lasty = x, y
					plt.plot(times, cum, label=f"agent_{i}")
				plt.xlabel("time_step")
				plt.ylabel("cumulative_distance")
				plt.title("Agent cumulative movement distance over time")
				plt.legend(loc="upper left")
				fname = os.path.join(run_run_dir, "agent_distance_over_time.png")
				plt.tight_layout()
				plt.savefig(fname, dpi=200)
				plt.close()
				print(f"Saved plot: {fname}")

			# (c) served cumulative over time
			served_times = metrics.get("served_times", [])
			max_t = 0
			if paths:
				for p in paths:
					if p:
						max_t = max(max_t, int(p[-1][0]))
			if served_times and max_t > 0:
				import numpy as _np
				bins = list(range(0, max_t + 2))
				hist, _ = _np.histogram(served_times, bins=bins)
				cum = _np.cumsum(hist)
				plt.figure(figsize=(10, 6))
				plt.plot(range(len(cum)), cum, marker="o")
				plt.xlabel("time_step")
				plt.ylabel("served_requests_cumulative")
				plt.title("Served demand (cumulative) over time")
				fname = os.path.join(run_run_dir, "served_over_time.png")
				plt.tight_layout()
				plt.savefig(fname, dpi=200)
				plt.close()
				print(f"Saved plot: {fname}")

		except Exception as e:
			print("Failed to save run outputs:", e)


def main() -> None:
	# --------------------------------------------------------------
	# Simplified CLI: auto-detect planner from checkpoint args
	# Priority: model (if checkpoints) > rule-mode > greedy (default)
	# --------------------------------------------------------------
	parser = argparse.ArgumentParser(description="DVRP runner")
	parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
	parser.add_argument("--render", action="store_true", help="Use pygame to visualize")
	parser.add_argument("--fps", type=int, default=10, help="Render FPS when --render (default: 10)")
	parser.add_argument("--save-run", action="store_true", help="Save per-run metrics and plots into outputs/run/<ts>/")
	# Rule-based planner mode (greedy or optimize)
	parser.add_argument("--rule-mode", choices=["greedy", "optimize"], default=None, 
						help="Use rule-based planner with specified mode (greedy/optimize)")
	parser.add_argument("--gmodel", action="store_true", help="Use neural net demand generator; otherwise rule")
	parser.add_argument("--service-time", action="store_true", help="Enable service times for demands (vehicles must remain on-site before completion)")
	parser.add_argument("--num-agents", type=int, default=2, help="Override number of agents for the episode (overrides config)")
	parser.add_argument("--map-wid", type=int, default=None, help="Override map width")
	parser.add_argument("--map-hei", type=int, default=None, help="Override map height")
	parser.add_argument("--total-demand", type=int, default=None, help="Override total demand parameter for the generator")
	parser.add_argument("--static-demands", action="store_true", help="Release all demands at time 0 to visualize static VRP instances")
	parser.add_argument("--static-max-end", type=int, default=None, help="Max environment time for static demands (default: 2 * max_time)")
	parser.add_argument("--max-steps", type=int, default=None, help="Maximum episode steps (default: no limit)")
	parser.add_argument("--static-ckpt", type=str, default=None, help="Path to V2 static model checkpoint (enables model planner)")
	parser.add_argument("--adapter-ckpt", type=str, default=None, help="Path to V2 dynamic adapter checkpoint (enables dynamic mode)")
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

	# Override max_end_time for static demands if specified
	if args.static_max_end is not None and args.static_max_end > 0:
		cfg.max_end_time = int(args.static_max_end)

	# Auto-detect planner based on checkpoint arguments
	# Priority: model (with checkpoints) > rule-mode > greedy (default)
	planner_choice: str
	planner_kwargs: Dict[str, Any] = {}
	
	if args.static_ckpt:
		# Model planner: static or dynamic based on adapter presence
		if args.adapter_ckpt:
			planner_choice = "dynamic"
			print(f"[AUTO] Using dynamic model (static + adapter)")
		else:
			planner_choice = "static"
			print(f"[AUTO] Using static model only")
		# Set checkpoint paths
		cfg.v2_planner_params = cfg.v2_planner_params if hasattr(cfg, 'v2_planner_params') else {}
		cfg.v2_planner_params["static_ckpt"] = args.static_ckpt
		if args.adapter_ckpt:
			cfg.v2_planner_params["adapter_ckpt"] = args.adapter_ckpt
	elif args.rule_mode:
		# Explicit rule-based planner mode
		planner_choice = args.rule_mode
		planner_kwargs["mode"] = args.rule_mode
		print(f"[AUTO] Using rule-based planner ({args.rule_mode})")
	else:
		# Default: greedy
		planner_choice = "greedy"
		planner_kwargs["mode"] = "greedy"
		print(f"[AUTO] Using default greedy planner")
	
	# Set config planner_type
	cfg.planner_type = planner_choice

	# Generator: net if --gmodel else keep default (rule)
	if args.gmodel:
		cfg.generator_type = "net"

	print(f"static_demands: {args.static_demands}")
	# Run
	run_episode(
		cfg,
		seed=args.seed,
		render=args.render,
		fps=args.fps,
		planner=planner_choice,
		static_demands=args.static_demands,
		planner_kwargs=planner_kwargs,
		save_run=bool(getattr(args, 'save_run', False)),
		max_steps=args.max_steps,
	)


if __name__ == "__main__":
	main()