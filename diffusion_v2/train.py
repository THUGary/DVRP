"""
train_adversarial_cycle: 主训练循环

流程:
1. DDIM 生成 VRP 实例
2. 物理斥力修复重叠
3. 坐标/需求缩放与裁剪
4. 合并重复节点（去重）
5. 环境打分 (Greedy + Baseline)
6. PPO 更新 Diffusion 模型

支持:
- 随机 Depot 位置 (RANDOMIZE_DEPOT)
- Depot-Aware 位置编码
- 完整后处理流程（sampler.generate 返回可直接使用的数据）
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diffusion_v2.model import (
    VRPDiffusionPolicy,
    create_global_condition,
    NUM_DIFFUSION_STEPS,
)
from diffusion_v2.sampler import InferenceSampler
from diffusion_v2.env import VRPGeneratorEnv
from diffusion_v2.ppo import PPOAgent

# ==============================================================================
# 训练超参数 (静态常量)
# ==============================================================================

# 训练参数
DEFAULT_NUM_EPOCHS: int = 1000        # 训练轮数
DEFAULT_EPISODES_PER_EPOCH: int = 32  # 每轮 episode 数
DEFAULT_EVAL_INTERVAL: int = 10       # 评估间隔
DEFAULT_SAVE_INTERVAL: int = 50       # 保存间隔

# VRP 问题参数
DEFAULT_NUM_NODES: int = 20           # 节点数
DEFAULT_MAP_SIZE: int = 30            # 地图大小
DEFAULT_TOTAL_DEMAND: int = 60        # 总需求量
DEFAULT_MAX_C: int = 10               # 最大单点需求
DEFAULT_CAPACITY: int = 30            # 车辆容量

# 默认 Depot 位置
DEFAULT_DEPOT_X: int = 15
DEFAULT_DEPOT_Y: int = 15

# 随机 Depot 设置
DEFAULT_RANDOMIZE_DEPOT: bool = False  # 是否随机化 depot 位置


def sample_random_depot(map_size: int) -> Tuple[int, int]:
    """
    随机采样 depot 位置
    
    Args:
        map_size: 地图大小
        
    Returns:
        (depot_x, depot_y): depot 坐标
    """
    depot_x = random.randint(0, map_size - 1)
    depot_y = random.randint(0, map_size - 1)
    return depot_x, depot_y


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train VRP Diffusion Generator with PPO"
    )
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--episodes-per-epoch", type=int, default=DEFAULT_EPISODES_PER_EPOCH,
                        help="Episodes per epoch")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # VRP 问题参数
    parser.add_argument("--num-nodes", type=int, default=DEFAULT_NUM_NODES,
                        help="Number of customer nodes")
    parser.add_argument("--map-size", type=int, default=DEFAULT_MAP_SIZE,
                        help="Map size (map-size x map-size)")
    parser.add_argument("--total-demand", type=int, default=DEFAULT_TOTAL_DEMAND,
                        help="Total demand to distribute")
    parser.add_argument("--max-c", type=int, default=DEFAULT_MAX_C,
                        help="Maximum demand per node")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY,
                        help="Vehicle capacity")
    
    # Depot 位置
    parser.add_argument("--depot-x", type=int, default=DEFAULT_DEPOT_X,
                        help="Depot X coordinate (used when randomize-depot is False)")
    parser.add_argument("--depot-y", type=int, default=DEFAULT_DEPOT_Y,
                        help="Depot Y coordinate (used when randomize-depot is False)")
    parser.add_argument("--randomize-depot", action="store_true", default=DEFAULT_RANDOMIZE_DEPOT,
                        help="Randomize depot position for each episode")
    
    # 采样参数
    parser.add_argument("--ddim-steps", type=int, default=10,
                        help="DDIM sampling steps")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="checkpoints/diffusion_v2",
                        help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/diffusion_v2",
                        help="Log directory for tensorboard")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL,
                        help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL,
                        help="Checkpoint save interval")
    
    # 硬件
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    
    # 加载检查点
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    return parser.parse_args()


def train_adversarial_cycle(
    model: VRPDiffusionPolicy,
    sampler: InferenceSampler,
    env: VRPGeneratorEnv,
    agent: PPOAgent,
    args: argparse.Namespace,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """
    主训练循环
    
    流程:
    1. DDIM 生成 -> 物理斥力修复 -> 环境打分 -> PPO 更新
    
    支持随机 Depot 位置，让模型学习不同 depot 下的需求分布
    
    Args:
        model: Diffusion 模型
        sampler: 采样器
        env: 环境
        agent: PPO 代理
        args: 命令行参数
        writer: TensorBoard writer
        
    Returns:
        metrics: 训练指标
    """
    device = next(model.parameters()).device
    
    # 固定 depot (仅在不随机化时使用)
    fixed_depot = (args.depot_x, args.depot_y)
    fixed_depot_norm_x = args.depot_x / (args.map_size - 1)
    fixed_depot_norm_y = args.depot_y / (args.map_size - 1)
    
    best_mean_reward = float('-inf')
    
    print(f"\n{'='*60}")
    print(f"Starting Adversarial Training")
    print(f"{'='*60}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Episodes per epoch: {args.episodes_per_epoch}")
    print(f"  Num nodes: {args.num_nodes}")
    print(f"  Map size: {args.map_size}")
    print(f"  Total demand: {args.total_demand}")
    print(f"  Max C: {args.max_c}")
    print(f"  Capacity: {args.capacity}")
    print(f"  Randomize Depot: {args.randomize_depot}")
    if not args.randomize_depot:
        print(f"  Fixed Depot: {fixed_depot}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_rewards = []
        epoch_greedy_lengths = []
        epoch_baseline_lengths = []  # 新增: baseline 长度
        epoch_regrets = []
        epoch_regrets_scaled = []  # 新增: 缩放后的 regret
        epoch_entropies = []
        epoch_distance_bonuses = []  # 新增: 距离奖励
        epoch_overlaps = 0
        epoch_overlap_ratios = []  # 新增: 合并比率列表
        
        # 收集 rollout
        for ep in range(args.episodes_per_epoch):
            # 决定 depot 位置
            if args.randomize_depot:
                depot_x, depot_y = sample_random_depot(args.map_size)
                depot = (depot_x, depot_y)
                depot_norm_x = depot_x / (args.map_size - 1)
                depot_norm_y = depot_y / (args.map_size - 1)
            else:
                depot = fixed_depot
                depot_norm_x = fixed_depot_norm_x
                depot_norm_y = fixed_depot_norm_y
            
            # 创建全局条件 (depot 位置已归一化)
            global_cond = create_global_condition(
                depot_x=depot_norm_x,
                depot_y=depot_norm_y,
                total_demand=args.total_demand,
                capacity=args.capacity,
                batch_size=1,
                device=device,
            )
            
            # 1. DDIM 生成（包含完整后处理：去重叠 + 缩放 + 裁剪 + 合并）
            coords, demands, info = sampler.generate(
                global_condition=global_cond,
                num_nodes=args.num_nodes,
                map_size=args.map_size,
                total_demand=args.total_demand,
                max_c=args.max_c,
                batch_size=1,
                apply_physics=True,
                merge_duplicates=True,  # 合并重复节点
            )
            
            # 检查物理斥力阶段的重叠
            has_overlap = info.get("has_overlap", False)
            if has_overlap:
                epoch_overlaps += 1
            
            # 获取合并信息
            overlap_ratios = info.get("overlap_ratios", [0.0])
            overlap_ratio = overlap_ratios[0] if overlap_ratios else 0.0
            
            # 2. 环境打分（传入已去重的坐标和需求）
            # 注意：env.get_reward 会再次检查并合并（如果有遗漏），但正常情况下已经合并完成
            reward, metrics = env.get_reward(
                coords=coords.squeeze(0),
                demands=demands.squeeze(0),
                depot=depot,  # 使用当前 episode 的 depot
                has_overlap=False,  # sampler 已经处理完成
            )
            
            epoch_rewards.append(reward)
            epoch_greedy_lengths.append(metrics["greedy_length"])
            epoch_baseline_lengths.append(metrics["baseline_length"])  # 新增
            epoch_regrets.append(metrics["regret"])
            epoch_regrets_scaled.append(metrics["regret_scaled"])  # 新增
            epoch_entropies.append(metrics["entropy"])
            epoch_distance_bonuses.append(metrics["distance_bonus"])  # 新增
            epoch_overlap_ratios.append(overlap_ratio)  # 使用 sampler 返回的合并比率
            
            # 3. 收集到 buffer
            # 为 PPO 生成伪输入 (使用采样过程中的最后一步)
            noisy_state = torch.randn(1, args.num_nodes, 3, device=device)
            timestep = torch.zeros(1, device=device, dtype=torch.long)
            
            agent.collect_rollout(
                noisy_state=noisy_state,
                timestep=timestep,
                global_condition=global_cond,
                reward=reward,
            )
        
        # 4. PPO 更新
        update_metrics = agent.update()
        
        # 计算统计
        mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        mean_greedy_len = sum(epoch_greedy_lengths) / len(epoch_greedy_lengths)
        mean_baseline_len = sum(epoch_baseline_lengths) / len(epoch_baseline_lengths)  # 新增
        mean_regret = sum(epoch_regrets) / len(epoch_regrets)
        mean_regret_scaled = sum(epoch_regrets_scaled) / len(epoch_regrets_scaled)  # 新增
        mean_entropy = sum(epoch_entropies) / len(epoch_entropies)
        mean_distance_bonus = sum(epoch_distance_bonuses) / len(epoch_distance_bonuses)  # 新增
        overlap_rate = epoch_overlaps / args.episodes_per_epoch
        mean_overlap_ratio = sum(epoch_overlap_ratios) / len(epoch_overlap_ratios)
        
        epoch_time = time.time() - epoch_start
        
        # 记录到 TensorBoard
        writer.add_scalar("train/reward", mean_reward, epoch)
        writer.add_scalar("train/greedy_length", mean_greedy_len, epoch)
        writer.add_scalar("train/baseline_length", mean_baseline_len, epoch)  # 新增
        writer.add_scalar("train/regret", mean_regret, epoch)
        writer.add_scalar("train/regret_scaled", mean_regret_scaled, epoch)  # 新增
        writer.add_scalar("train/entropy", mean_entropy, epoch)
        writer.add_scalar("train/distance_bonus", mean_distance_bonus, epoch)  # 新增
        writer.add_scalar("train/overlap_rate", overlap_rate, epoch)
        writer.add_scalar("train/overlap_ratio", mean_overlap_ratio, epoch)
        writer.add_scalar("train/policy_loss", update_metrics["policy_loss"], epoch)
        writer.add_scalar("train/ppo_entropy", update_metrics["entropy"], epoch)
        
        # 打印进度（每个 epoch 都打印，增加 baseline 显示）
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"Reward: {mean_reward:7.4f} | "
            f"Greedy: {mean_greedy_len:7.2f} | "
            f"Base: {mean_baseline_len:6.2f} | "  # 新增: baseline
            f"Regret: {mean_regret:6.4f} | "
            f"Entropy: {mean_entropy:5.3f} | "
            f"Merged: {mean_overlap_ratio:.1%} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # 保存最佳模型
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_path = os.path.join(args.output_dir, "best.pth")
            agent.save(best_path)
            print(f"  ✓ New best reward: {best_mean_reward:.4f}, saved to {best_path}")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
            agent.save(ckpt_path)
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final.pth")
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    return {
        "best_reward": best_mean_reward,
        "final_reward": mean_reward,
    }


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    args.log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # 创建模型
    model = VRPDiffusionPolicy().to(device)
    print(f"Created VRPDiffusionPolicy with {model.get_num_params():,} parameters")
    
    # 创建采样器
    sampler = InferenceSampler(model, ddim_steps=args.ddim_steps)
    
    # 创建环境
    env = VRPGeneratorEnv(map_size=args.map_size, capacity=args.capacity)
    
    # 创建 PPO 代理
    agent = PPOAgent(model, lr=args.lr)
    
    # 加载检查点 (如果指定)
    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)
    
    # 训练
    try:
        metrics = train_adversarial_cycle(
            model=model,
            sampler=sampler,
            env=env,
            agent=agent,
            args=args,
            writer=writer,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        # 保存中断时的模型
        interrupt_path = os.path.join(args.output_dir, "interrupted.pth")
        agent.save(interrupt_path)
        print(f"Saved interrupted model to {interrupt_path}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
