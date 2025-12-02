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
from agent.planner import ModelPlanner
from agent.planner import CVRPPOMOPlanner
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


def run_episode(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, planner: str = "greedy", static_demands: bool = False) -> None:
	print(f"depot: {cfg.depot}, num_agents: {cfg.num_agents}, capacity: {cfg.capacity}, max_time: {cfg.max_time}")
	
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
	env, gen, planner_impl, controller = build_env(cfg, planner_type=planner_type,static_demands=static_demands)
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
		)

		# Plan targets using current observation with enhanced information
		agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
		plan_horizon = 1
		if planner_type == "model":
			plan_horizon = max(1, int(cfg.model_planner_params.get("time_plan", 1)))
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
			cfg.planner_type = "model"  # use defaults from cfg.model_planner_params
			if isinstance(args.pmodel, str) and args.pmodel != "__DEFAULT__":
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
				if ckpt_path is None:
					ckpt_path = os.path.abspath(raw)
					print(f"WARNING: Planner checkpoint not found at {ckpt_path}. Will attempt to load; ensure path is correct.")
				cfg.model_planner_params["ckpt"] = ckpt_path

	if args.gmodel:
		cfg.generator_type = "net"
	return cfg, planner_choice

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
	parser.add_argument("--gmodel", action="store_true", help="Use neural net demand generator; otherwise rule")
	parser.add_argument("--service-time", action="store_true", default=True, help="Enable service times for demands (vehicles must remain on-site before completion)")
	parser.add_argument("--cvrp", action="store_true", help="Use the CVRP-POMO planner adapter instead of DVRP planners")
	parser.add_argument("--cvrp-root", type=str, default=None, help="Override path to the CVRP/POMO folder (defaults to config)")
	parser.add_argument("--cvrp-ckpt", type=str, default=None, help="Override path to the CVRP checkpoint (.pt)")
	# parser.add_argument("--num-agents", type=int, default=2, help="Override number of agents for the episode (overrides config)")
	parser.add_argument("--test-all", action="store_true", help="Run all instances in the specified dataset (not implemented yet)")
	parser.add_argument("--instance", type=str, default="R101", help="Specify the problem instance name to run (default: R104)")
	parser.add_argument("--least-vehs", action="store_true", help="Use the least number of vehicles used in known solution for the instance")
	parser.add_argument("--static-demands", action="store_true", help="Use static demands (all appear at t=0)")
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