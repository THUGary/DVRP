from __future__ import annotations
import argparse
from typing import List, Tuple

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.generator import RuleBasedGenerator
from agent.planner import RuleBasedPlanner
from agent.controller import RuleBasedController
from utils.pygame_renderer import PygameRenderer


def build_env(cfg: Config) -> Tuple[GridEnvironment, RuleBasedGenerator, RuleBasedPlanner, RuleBasedController]:
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
	planner = RuleBasedPlanner(**cfg.planner_params)
	controller = RuleBasedController(**cfg.controller_params)
	return env, gen, planner, controller


def run_episode(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10) -> None:
	env, gen, planner, controller = build_env(cfg)
	obs = env.reset(seed)
	total_reward = 0.0
	done = False
	step = 0
	renderer = None
	if render:
		renderer = PygameRenderer(cfg.width, cfg.height)
		renderer.init()
	while not done:
		# Plan targets using current observation
		agent_states = obs["agent_states"]  # list of (x,y,s)
		agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]
		targets = planner.plan(
			observations=obs["demands"],
			agent_states=agents,  # uses x,y,s only
			depot=obs["depot"],
			t=obs["time"],
			horizon=1,
		)
		# Controller decides per-agent move
		actions: List[Tuple[int, int]] = []
		for i, (x, y, s) in enumerate(agent_states):
			actions.append(controller.act((x, y), targets[i]))
		obs, reward, done, info = env.step(actions)
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
			print(f"Step {step:03d} | time={obs['time']} | reward={reward:.0f} | total={total_reward:.0f} | demands={len(obs['demands'])}")
	print(f"Episode done in {step} steps. Total reward={total_reward:.0f}")
	if renderer is not None:
		renderer.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Train skeleton (rule-based run)")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--render", action="store_true", help="Use pygame to visualize")
	parser.add_argument("--fps", type=int, default=10, help="Render FPS when --render")
	args = parser.parse_args()
	cfg = get_default_config()
	run_episode(cfg, seed=args.seed, render=args.render, fps=args.fps)


if __name__ == "__main__":
	main()
