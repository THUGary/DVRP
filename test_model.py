from __future__ import annotations
import argparse
from typing import List, Tuple, Deque
from collections import deque

from configs import get_default_config, Config
from environment.env import GridEnvironment
from agent.generator import RuleBasedGenerator
from agent.controller import RuleBasedController
from agent.planner import ModelPlanner
from agent.planner.base import AgentState, Target
from utils.pygame_renderer import PygameRenderer


def build_env(cfg: Config):
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
    return env


def main():
    ap = argparse.ArgumentParser(description="Test trained DVRP planner model with environment")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoints/planner_{map_wid}_{agent_num}_{epoch}.pt")
    ap.add_argument("--map_wid", type=int, default=20)
    ap.add_argument("--map_hei", type=int, default=20)
    ap.add_argument("--agent_num", type=int, default=2)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--time_plan", type=int, default=3)
    args = ap.parse_args()

    cfg = get_default_config()
    cfg.width = args.map_wid
    cfg.height = args.map_hei
    cfg.num_agents = args.agent_num

    env = build_env(cfg)
    planner = ModelPlanner(time_plan=args.time_plan, device="cpu", full_capacity=cfg.capacity)
    # 需要 ModelPlanner 实现 load_from_ckpt(path)
    if hasattr(planner, "load_from_ckpt"):
        planner.load_from_ckpt(args.ckpt)
    controller = RuleBasedController(**cfg.controller_params)

    obs = env.reset(seed=0)
    renderer = None
    if args.render:
        renderer = PygameRenderer(cfg.width, cfg.height)
        renderer.init()

    total_reward = 0.0
    done = False
    step = 0
    current_plans: List[Deque[Tuple[int, int]]] = [deque() for _ in range(cfg.num_agents)]
    prev_demands = []

    while not done:
        demands = obs["demands"]
        new_demands = [d for d in demands if d not in prev_demands]
        agent_states = obs["agent_states"]

        # 构造 AgentState
        agents = [AgentState(x=x, y=y, s=s) for (x, y, s) in agent_states]

        # 触发逻辑：有新需求或某 agent 无后续目标 -> 重规划
        can_continue = all(len(q) > 0 for q in current_plans)
        need_replan = (len(new_demands) > 0) or (not can_continue)

        if need_replan:
            targets = planner.plan(
                observations=[(x, y, t, c, end_t) for (x, y, t, c, end_t) in demands],
                agent_states=agents,
                depot=obs["depot"],
                t=obs["time"],
                current_plans=current_plans,
                horizon=max(1, int(args.time_plan)),
            )
            current_plans = targets

        # 执行动作：朝队首目标移动一步
        actions: List[Tuple[int, int]] = []
        for i, (x, y, s) in enumerate(agent_states):
            # 弹掉已到达
            while len(current_plans[i]) > 0 and current_plans[i][0] == (x, y):
                current_plans[i].popleft()
            if len(current_plans[i]) == 0:
                actions.append((0, 0))
            else:
                actions.append(controller.act((x, y), current_plans[i]))

        obs, reward, done, info = env.step(actions)
        prev_demands = list(demands)

        if renderer is not None:
            if not renderer.render(obs):
                break
            if args.fps > 0:
                import time as _t
                _t.sleep(1.0 / args.fps)

        total_reward += reward
        step += 1
        if step % 10 == 0 or done:
            print(f"Step {step:03d} | t={obs['time']} | reward={reward:.0f} | total={total_reward:.0f} | demands={len(obs['demands'])}")

    if renderer is not None:
        renderer.close()
    print(f"Episode done in {step} steps. Total reward={total_reward:.0f}")


if __name__ == "__main__":
    main()