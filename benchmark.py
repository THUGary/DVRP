from __future__ import annotations
import argparse
import random
import os
import copy
from typing import List, Tuple
import pandas as pd

from configs import Config
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
from agent.generator.benchmark_gen import BenchmarkGenerator

def get_benchmark_config(dataset_basepath: str, problem_type: str,instance_info: dict, least_vehicles: bool) -> Config:
	"""
	* **dataset_basepath**: the base path where the problem instances of any type of\n 
	`problem_type` are stored.
	* **problem_type**: one of the following\n
	`solomon, homberger_200, homberger_400, homberger_600, homberger_800 and homberger_1000`\n
	* **instance_info**: includes information such as\n 
	`problem_name, vehicle_number, vehicle_capacity, depot_x, depot_y, duration`(necessary)\n
	`total_customers, total_demand` (optional).
	* **least_vehicles**: If *True*, use the least number of vehicles used in known solution.\n
	If *False*, use the maximum allowed number of vehicles specified in instance_info.
	"""
	# Check if instance_info has all necessary keys with non-None values``
	list=['problem_name','vehicle_number','vehicle_capacity','depot_x','depot_y', 'duration']
	for key in list:
		if key not in instance_info:
			raise ValueError(f"{key} is missing in instance_info.")
		elif instance_info[key] is None:
			raise ValueError(f"{key} value is None in instance_info.")
		
	if not os.path.isdir(dataset_basepath):
		raise ValueError(f"Dataset basepath {dataset_basepath} does not exist or is not a directory.")
	problem_name=instance_info['problem_name']
	customer_path = os.path.join(dataset_basepath, f'customers/{problem_name}_customers.csv')
	df=pd.read_csv(customer_path)
	
	solution_summary_path = os.path.join(dataset_basepath, 'solution_summary.csv')
	solution_summary=pd.read_csv(solution_summary_path)
	solution_info=solution_summary[solution_summary['problem_name']==problem_name]
	least_num_vehicles= solution_info['vehicle_number'].values[0]
	print(f"Least number of vehicles used in known solution: {least_num_vehicles}")

	env_map_size={
		"solomon": (100,100),
		"homberger_200": (140,140),
		"homberger_400": (200,200),
		"homberger_600": (300,300),
		"homberger_800": (400,400),
		"homberger_1000": (500,500),
	}
	
	map_size= env_map_size.get(problem_type, (100,100))
	
	config_params={
		"height":map_size[0],
		"width":map_size[1],
		"num_agents":least_num_vehicles if least_vehicles else instance_info.get("vehicle_number"),
		"capacity":instance_info.get("vehicle_capacity"),
		"depot":(instance_info.get("depot_x"), instance_info.get("depot_y")),
		"max_time":instance_info.get("duration")+100,  # extra time buffer
		"generator_type":"benchmark",
		"generator_params":{
			"instance_data": df,
			"max_time": instance_info.get("duration")+100,
		}
	}
	
	# Only part of the config data is used for benchmark,
	# other parameters are same as defaulted in Config class
	return Config(**config_params)


def build_env(cfg: Config, planner_type: str, static_demands: bool) -> Tuple[GridEnvironment, BaseDemandGenerator, BasePlanner, RuleBasedController]:
	# choose the BenchmarkGenerator
	if static_demands:
		from agent.generator.static_benchmark_gen import BenchmarkGenerator
	else:
		from agent.generator.benchmark_gen import BenchmarkGenerator
	gen = BenchmarkGenerator(cfg.width, cfg.height, **cfg.generator_params)
	
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
		max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
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
	elif planner_type in ("model", "static", "dynamic"):
		# Use V2Planner for model/static/dynamic
		v2_params = dict(getattr(cfg, 'v2_planner_params', {}))
		# Map "model" to "dynamic" mode
		mode = "dynamic" if planner_type in ("model", "dynamic") else "static"
		planner = create_v2_planner(
			mode=mode,
			grid_width=cfg.width,
			grid_height=cfg.height,
			full_capacity=cfg.capacity,
			**v2_params,
		)
	else:
		raise ValueError(f"Unknown planner type: {planner_type}")
	controller = RuleBasedController(**cfg.controller_params)
	return env, gen, planner, controller


def run_episode(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, planner: str = "greedy", static_demands: bool = False) -> None:
	print(f"depot: {cfg.depot}, num_agents: {cfg.num_agents}, capacity: {cfg.capacity}, max_time: {cfg.max_time}")
	
	#print model used
	print(f"Using planner: {planner}")
	planner_type = planner
	if planner_type in ("model", "static", "dynamic"):
		v2_params = getattr(cfg, 'v2_planner_params', {})
		print(f"V2Planner mode: {'dynamic' if planner_type in ('model', 'dynamic') else 'static'}")
		if v2_params.get("static_ckpt"):
			print(f"Static checkpoint: {v2_params['static_ckpt']}")
		if v2_params.get("adapter_ckpt"):
			print(f"Adapter checkpoint: {v2_params['adapter_ckpt']}")
	env, gen, planner_impl, controller = build_env(cfg, planner_type=planner_type)
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

	if render:
		renderer = PygameRenderer(cfg.width, cfg.height, cell_size=10)
		renderer.init()

	while not done:
		# 检测新增的需求
		current_demands = obs["demands"]
		new_demands = [d for d in current_demands if d not in prev_demands]

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
		if planner_type in ("model", "dynamic"):
			plan_horizon = max(1, int(getattr(cfg, 'v2_planner_params', {}).get("time_plan", 1)))
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
			print(f"Step {step:03d} | time={obs['time']} | reward={reward:.0f} | total={total_reward:.0f} | demands={len(obs['demands'])} | unserved={unserved}")
	print(f"Episode done in {step} steps. Total reward={total_reward:.0f}")
	if renderer is not None:
		renderer.close()


def gen_plan_choice(args: argparse.Namespace, cfg)->str:
	"""Simplified planner selection for V2Planner architecture"""
	planner_choice = getattr(args, 'planner', 'greedy')
	
	# Map "model" to "dynamic" for backwards compatibility  
	if planner_choice == "model":
		planner_choice = "dynamic"
	
	cfg.planner_type = planner_choice
	
	# Handle V2Planner checkpoint overrides
	if planner_choice in ("static", "dynamic"):
		if not hasattr(cfg, 'v2_planner_params'):
			cfg.v2_planner_params = {}
		if hasattr(args, 'static_ckpt') and args.static_ckpt:
			cfg.v2_planner_params["static_ckpt"] = args.static_ckpt
		if hasattr(args, 'adapter_ckpt') and args.adapter_ckpt and planner_choice == "dynamic":
			cfg.v2_planner_params["adapter_ckpt"] = args.adapter_ckpt

	if args.gmodel:
		cfg.generator_type = "net"
	return cfg, planner_choice

def main() -> None:
	# --------------------------------------------------------------
	# Minimal CLI for benchmark evaluation with V2Planner
	# --------------------------------------------------------------
	parser = argparse.ArgumentParser(description="DVRP benchmark runner")
	parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025)")
	parser.add_argument("--render", action="store_true", help="Use pygame to visualize")
	parser.add_argument("--fps", type=int, default=10, help="Render FPS when --render (default: 10)")
	parser.add_argument("--planner", choices=["greedy", "model", "static", "dynamic", "fri", "rbso", "dcp"], 
						default="greedy", help="Planner type: greedy (rule-based), model/static/dynamic (V2Planner), fri, rbso, dcp")
	parser.add_argument("--gmodel", action="store_true", help="Use neural net demand generator; otherwise rule")
	parser.add_argument("--service-time", action="store_true", default=True, help="Enable service times for demands (vehicles must remain on-site before completion)")
	parser.add_argument("--test-all", action="store_true", help="Run all instances in the specified dataset (not implemented yet)")
	parser.add_argument("--instance", type=str, default="R101", help="Specify the problem instance name to run (default: R104)")
	parser.add_argument("--least-vehs", action="store_true", help="Use the least number of vehicles used in known solution for the instance")
	parser.add_argument("--static-ckpt", type=str, default=None, help="Override path to V2 static model checkpoint")
	parser.add_argument("--adapter-ckpt", type=str, default=None, help="Override path to V2 dynamic adapter checkpoint")
	args = parser.parse_args()

	dataset_basepath = "./VrptwDataset/solomon_reformed"  # specify your dataset base path here
	problem_type = "solomon"  # specify your problem type here
	problem_index_path=os.path.join(dataset_basepath,"Problem_Index.csv")
	problem_index=pd.read_csv(problem_index_path)

	least_vehicles = args.least_vehs
	static_demands = args.static_demands
	
	if args.test_all:
		problem_names=problem_index['problem_name'].tolist()
		for pname in problem_names:
			print(f"Running instance: {pname}")
			instance_info=problem_index[problem_index['problem_name']==pname].iloc[0].to_dict()
			cfg = get_benchmark_config(dataset_basepath, problem_type, instance_info, least_vehicles)
			cfg.include_service_time = bool(args.service_time)
			print(f"Service time enabled: {cfg.include_service_time}")
			cfg, planner_choice = gen_plan_choice(args, cfg)
			# Run, here the seed does need to be set
			run_episode(cfg, seed=args.seed, render=args.render, 
			   fps=args.fps, planner=planner_choice, static_demands=static_demands)
	else:
		problem_name = args.instance  # specify your problem instance name in the arguments first
		print(f"Running instance: {problem_name}")
		instance_info=problem_index[problem_index['problem_name']==problem_name]

		if instance_info.empty:
			raise ValueError(f"Problem instance {problem_name} not found in index.")
		instance_info=instance_info.iloc[0].to_dict()

		cfg = get_benchmark_config(dataset_basepath, problem_type, instance_info, least_vehicles)
		cfg.include_service_time = bool(args.service_time)
		cfg, planner_choice = gen_plan_choice(args, cfg)
		run_episode(cfg, seed=args.seed, render=args.render, 
			   fps=args.fps, planner=planner_choice, static_demands=static_demands)
	


if __name__ == "__main__":
	main()