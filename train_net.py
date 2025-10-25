from __future__ import annotations
import argparse
from typing import List, Tuple, Deque
from collections import deque

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.generator import RuleBasedGenerator
from agent.controller import RuleBasedController
from agent.planner.model_planner import ModelPlanner
from utils.pygame_renderer import PygameRenderer


def build_env(cfg: Config) -> Tuple[GridEnvironment, RuleBasedGenerator, ModelPlanner, RuleBasedController]:
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
    planner = ModelPlanner(d_model=128, nhead=8, nlayers=2, time_plan=6, lateness_lambda=0.0, device="cpu")
    controller = RuleBasedController(**cfg.controller_params)
    return env, gen, planner, controller


def run_episode(cfg: Config, seed: int = 0, render: bool = False, fps: int = 10, time_plan: int = 6) -> None:
    env, gen, planner, controller = build_env(cfg)
    planner.time_plan = time_plan

    obs = env.reset(seed)
    total_reward = 0.0
    done = False
    step = 0
    renderer = None
    if render:
        renderer = PygameRenderer(cfg.width, cfg.height)
        renderer.init()

    # 维护当前规划（目标队列）
    current_plans: List[Deque[Tuple[int, int]]] = [deque() for _ in range(cfg.num_agents)]
    prev_demands = []

    while not done:
        # 判断是否需要重规划：出现新节点 或 有agent没有后续目标
        demands = obs["demands"]  # [(x,y,t,c,end_t)]
        new_demands = [d for d in demands if d not in prev_demands]
        need_replan = len(new_demands) > 0 or any(len(q) == 0 for q in current_plans)

        # 准备 agent_states 结构用于 planner：[(x,y,s)]
        agent_states = obs["agent_states"]
        agents = [type("S", (), {"x": x, "y": y, "s": s}) for (x, y, s) in agent_states]

        if need_replan:
            # planner.plan：模型编码一次，顺序解码多轮得到每个 agent 在未来 time_plan 内的路径队列
            targets = planner.plan(
                observations=[(x, y, t, c, end_t) for (x, y, t, c, end_t) in demands],
                agent_states=agents,
                depot=obs["depot"],
                t=obs["time"],
                current_plans=current_plans,
            )
            current_plans = targets  # 覆盖为最新规划

        # 基于当前计划出一个 step 的 actions（每个 agent 向自己的队列头移动一步）
        actions: List[Tuple[int, int]] = []
        for i, (x, y, s) in enumerate(agent_states):
            # 若队列为空或队首为当前位置，则弹出
            while len(current_plans[i]) > 0 and current_plans[i][0] == (x, y):
                current_plans[i].popleft()
            if len(current_plans[i]) == 0:
                actions.append((0, 0))
            else:
                target_xy = current_plans[i][0]
                actions.append(controller.act((x, y), deque([target_xy])))

        obs, reward, done, info = env.step(actions)
        prev_demands = list(demands)

        if renderer is not None:
            if not renderer.render(obs):
                break
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
    parser = argparse.ArgumentParser(description="Train DVRP with model-based planner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Use pygame to visualize")
    parser.add_argument("--fps", type=int, default=10, help="Render FPS when --render")
    parser.add_argument("--time_plan", type=int, default=10, help="Plan horizon in time units for each replan")
    args = parser.parse_args()

    cfg = get_default_config()
    run_episode(cfg, seed=args.seed, render=args.render, fps=args.fps, time_plan=args.time_plan)


if __name__ == "__main__":
    main()