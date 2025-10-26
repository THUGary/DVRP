from __future__ import annotations
import argparse
from typing import List, Tuple, Dict

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.generator import RuleBasedGenerator
from agent.controller import RuleBasedController
from utils.pygame_renderer import PygameRenderer
from utils.state_manager import PlanningState, update_planning_state

from agent.planner import RuleBasedPlanner
from agent.planner import FastReactiveInserter
from agent.planner import RepairBasedStabilityOptimizer
from agent.planner import DistributedCooperativePlanner

def _palette(n: int):
	base = [
		(255, 179, 186),  # soft pink
		(255, 223, 186),  # peach
		(255, 255, 186),  # light yellow
		(186, 255, 201),  # mint green
		(186, 225, 255),  # sky blue
		(220, 186, 255),  # lavender
		(255, 206, 237),  # light rose
		(191, 255, 249),  # pale aqua
		(255, 244, 209),  # cream
		(232, 243, 255),  # very light blue
	]
	return [base[i % len(base)] for i in range(n)]

def build_env(cfg: Config, planner_type: str) -> Tuple[GridEnvironment, RuleBasedGenerator, RuleBasedPlanner, RuleBasedController]:
	gen = RuleBasedGenerator(cfg.width, cfg.height, **cfg.generator_params)
	env = GridEnvironment(
		width=cfg.width,
		height=cfg.height,
		num_agents=cfg.num_agents,
		capacity=cfg.capacity,
		depot=cfg.depot,
		generator=gen,
		max_time=cfg.max_time,
	)
	env.num_agents = cfg.num_agents
	if planner_type == "greedy":
		# 使用 Rule-based Planner
		planner = RuleBasedPlanner(**cfg.planner_params)
	elif planner_type == "fri":
		# 使用 Fast Reactive Inserter
		planner = FastReactiveInserter()
	elif planner_type == "rbso":
		# 使用 Repair-based Stability Optimizer（带参数）
		planner = RepairBasedStabilityOptimizer(destroy_ratio=0.3, local_search_iters=10)
	elif planner_type == "dcp":
		# 使用 Distributed Cooperative Planner（带参数）
		planner = DistributedCooperativePlanner(auction_rounds=5, bid_strategy='time_urgency')
	else:
		raise ValueError(f"Unknown planner type: {planner_type}")

	controller = RuleBasedController(**cfg.controller_params)
	return env, gen, planner, controller


def run_episode(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, planner: str = "greedy") -> None:
	env, gen, planner, controller = build_env(cfg, planner_type=planner)
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
		renderer = PygameRenderer(cfg.width, cfg.height)
		renderer.init()
	colors = _palette(cfg.num_agents)
	
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
		targets = planner.plan(
			observations=obs["demands"],  # [(x, y, t_arrival, c, t_due), ...]
			agent_states=agents,
			depot=obs["depot"],
			t=obs["time"],
			horizon=1,
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
			# 构造 planned_tasks 以对齐渲染接口
			planned_tasks: Dict[int, List[Tuple[int,int]]] = {i: list(targets[i]) for i in range(cfg.num_agents)}
			if not renderer.render(obs, agent_colors=colors, planned_tasks=planned_tasks):
				break
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Train skeleton (rule-based run)")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--render", action="store_true", help="Use pygame to visualize")
	parser.add_argument("--fps", type=int, default=10, help="Render FPS when --render")
	parser.add_argument("--planner", type=str, default="greedy", help="Planner type: greedy, fri, rbso, dcp")
	args = parser.parse_args()
	cfg = get_default_config()
	run_episode(cfg, seed=args.seed, render=args.render, fps=args.fps, planner=args.planner)


if __name__ == "__main__":
	main()