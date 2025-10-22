from __future__ import annotations
from configs import get_default_config
from train import run_episode


def smoke_test():
	cfg = get_default_config()
	# Shorten for quick test
	cfg.max_time = 10
	cfg.width = 6
	cfg.height = 6
	cfg.num_agents = 1
	run_episode(cfg, seed=42)


if __name__ == "__main__":
	smoke_test()
