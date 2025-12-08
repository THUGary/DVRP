"""
Diffusion V2: 全新的 VRP 需求生成 Diffusion 架构

模块结构:
- model.py: VRPDiffusionPolicy (Nano-DiT 架构)
- sampler.py: InferenceSampler (DDIM 采样 + 物理后处理)
- env.py: VRPGeneratorEnv (Greedy Planner 环境与奖励)
- ppo.py: PPOAgent (PPO 训练代理)
- train.py: train_adversarial_cycle (主训练循环)
- adapter.py: DiffusionV2Generator (兼容旧接口的适配器)
- visualize.py: 可视化工具
"""

from .model import VRPDiffusionPolicy, create_global_condition
from .sampler import InferenceSampler
from .env import VRPGeneratorEnv, GreedyPlanner
from .ppo import PPOAgent
from .adapter import DiffusionV2Generator, DemandGenerator

__all__ = [
    # 核心模型
    "VRPDiffusionPolicy",
    "create_global_condition",
    # 采样器
    "InferenceSampler",
    # 环境
    "VRPGeneratorEnv",
    "GreedyPlanner",
    # PPO
    "PPOAgent",
    # 适配器
    "DiffusionV2Generator",
    "DemandGenerator",
]

