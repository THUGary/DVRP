from __future__ import annotations
from configs import get_default_config
from train_net import run_episode


def smoke_test():
    cfg = get_default_config()
    cfg.max_time = 12
    cfg.width = 8
    cfg.height = 8
    cfg.num_agents = 2
    run_episode(cfg, seed=123, render=False, fps=0, time_plan=4)


if __name__ == "__main__":
    smoke_test()