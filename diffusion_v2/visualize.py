"""
Diffusion V2 可视化工具

功能:
- 可视化生成的 VRP 实例
- 显示坐标分布、需求分布
- 对比不同检查点的生成质量
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 添加项目根目录
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diffusion_v2.model import VRPDiffusionPolicy, create_global_condition
from diffusion_v2.sampler import InferenceSampler
from diffusion_v2.env import VRPGeneratorEnv


def visualize_vrp_instance(
    coords: np.ndarray,
    demands: np.ndarray,
    depot: Tuple[int, int],
    map_size: int,
    title: str = "VRP Instance",
    ax: Optional[plt.Axes] = None,
    show_demands: bool = True,
) -> plt.Axes:
    """
    可视化单个 VRP 实例
    
    Args:
        coords: (N, 2) 节点坐标
        demands: (N,) 节点需求
        depot: depot 坐标
        map_size: 地图大小
        title: 标题
        ax: matplotlib axes
        show_demands: 是否显示需求值
        
    Returns:
        ax: matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 绘制网格
    ax.set_xlim(-0.5, map_size - 0.5)
    ax.set_ylim(-0.5, map_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制节点
    # 根据需求大小调整点的大小
    max_demand = max(demands.max(), 1)
    sizes = 50 + 200 * (demands / max_demand)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes,
        c=demands,
        cmap='Reds',
        alpha=0.7,
        edgecolors='black',
        linewidths=1,
    )
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='Demand')
    
    # 标注需求值
    if show_demands:
        for i, (x, y) in enumerate(coords):
            ax.annotate(
                f'{int(demands[i])}',
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8,
            )
    
    # 绘制 depot
    ax.scatter(
        [depot[0]], [depot[1]],
        s=200,
        c='green',
        marker='s',
        edgecolors='black',
        linewidths=2,
        label='Depot',
        zorder=10,
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    
    return ax


def visualize_demand_distribution(
    demands: np.ndarray,
    title: str = "Demand Distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    可视化需求分布直方图
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.hist(demands, bins=range(0, int(demands.max()) + 2), 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Demand')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # 添加统计信息
    stats_text = f'Mean: {demands.mean():.2f}\nStd: {demands.std():.2f}\nTotal: {demands.sum()}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return ax


def visualize_spatial_heatmap(
    coords: np.ndarray,
    map_size: int,
    grid_size: int = 5,
    title: str = "Spatial Distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    可视化空间分布热图
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # 创建网格统计
    cell_size = map_size / grid_size
    heatmap = np.zeros((grid_size, grid_size))
    
    for x, y in coords:
        gx = min(int(x / cell_size), grid_size - 1)
        gy = min(int(y / cell_size), grid_size - 1)
        heatmap[gy, gx] += 1
    
    im = ax.imshow(heatmap, cmap='YlOrRd', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, label='Node Count')
    
    # 添加网格线
    for i in range(grid_size + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=0.5)
        ax.axvline(x=i - 0.5, color='black', linewidth=0.5)
    
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title(title)
    
    return ax


def generate_and_visualize(
    checkpoint_path: Optional[str],
    num_samples: int = 4,
    num_nodes: int = 20,
    map_size: int = 30,
    total_demand: int = 60,
    max_c: int = 10,
    capacity: int = 30,
    depot: Tuple[int, int] = (15, 15),
    ddim_steps: int = 10,
    save_path: Optional[str] = None,
    device: str = "auto",
):
    """
    生成 VRP 实例并可视化
    """
    # 设置设备
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # 创建模型
    model = VRPDiffusionPolicy().to(device)
    
    # 加载检查点
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Using randomly initialized model")
    
    # 创建采样器和环境
    sampler = InferenceSampler(model, ddim_steps=ddim_steps)
    env = VRPGeneratorEnv(map_size=map_size, capacity=capacity)
    
    # 归一化 depot
    depot_norm_x = depot[0] / (map_size - 1)
    depot_norm_y = depot[1] / (map_size - 1)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 4 * num_samples))
    
    all_coords = []
    all_demands = []
    all_metrics = []
    
    for i in range(num_samples):
        print(f"\nGenerating sample {i+1}/{num_samples}...")
        
        # 创建条件
        global_cond = create_global_condition(
            depot_x=depot_norm_x,
            depot_y=depot_norm_y,
            total_demand=total_demand,
            capacity=capacity,
            batch_size=1,
            device=device,
        )
        
        # 生成
        coords, demands, info = sampler.generate(
            global_condition=global_cond,
            num_nodes=num_nodes,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            batch_size=1,
        )
        
        coords_np = coords.squeeze(0).cpu().numpy()
        demands_np = demands.squeeze(0).cpu().numpy().flatten()
        
        all_coords.append(coords_np)
        all_demands.append(demands_np)
        
        # 计算指标
        reward, metrics = env.get_reward(
            coords.squeeze(0), demands.squeeze(0), depot, info.get("has_overlap", False)
        )
        all_metrics.append(metrics)
        
        print(f"  Reward: {reward:.4f}, Greedy: {metrics['greedy_length']:.2f}, "
              f"Regret: {metrics['regret']:.4f}, Entropy: {metrics['entropy']:.3f}")
        
        # 绘制
        ax1 = fig.add_subplot(num_samples, 4, i * 4 + 1)
        visualize_vrp_instance(
            coords_np, demands_np, depot, map_size,
            title=f"Sample {i+1} (Reward: {reward:.3f})", ax=ax1
        )
        
        ax2 = fig.add_subplot(num_samples, 4, i * 4 + 2)
        visualize_demand_distribution(demands_np, title=f"Demand Dist. {i+1}", ax=ax2)
        
        ax3 = fig.add_subplot(num_samples, 4, i * 4 + 3)
        visualize_spatial_heatmap(coords_np, map_size, title=f"Spatial Dist. {i+1}", ax=ax3)
        
        # 添加指标文本
        ax4 = fig.add_subplot(num_samples, 4, i * 4 + 4)
        ax4.axis('off')
        metrics_text = (
            f"Metrics for Sample {i+1}:\n"
            f"{'─' * 30}\n"
            f"Greedy Length: {metrics['greedy_length']:.2f}\n"
            f"Baseline Length: {metrics['baseline_length']:.2f}\n"
            f"Regret: {metrics['regret']:.4f}\n"
            f"Regret (scaled): {metrics.get('regret_scaled', metrics['regret']):.4f}\n"
            f"Spatial Entropy: {metrics['entropy']:.3f}\n"
            f"Distance Bonus: {metrics.get('distance_bonus', 0):.4f}\n"
            f"Valid Mask: {metrics['valid_mask']:.0f}\n"
            f"Num Routes: {metrics['num_routes']}\n"
            f"{'─' * 30}\n"
            f"Total Demand: {demands_np.sum()}\n"
            f"Max Demand: {demands_np.max()}\n"
            f"Has Overlap: {metrics.get('has_overlap', info.get('has_overlap', False))}\n"
        )
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                 verticalalignment='top', fontfamily='monospace',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
    else:
        plt.show()
    
    # 打印汇总统计
    print("\n" + "=" * 50)
    print("Summary Statistics")
    print("=" * 50)
    avg_reward = np.mean([m['reward'] for m in all_metrics])
    avg_greedy = np.mean([m['greedy_length'] for m in all_metrics])
    avg_regret = np.mean([m['regret'] for m in all_metrics])
    avg_regret_scaled = np.mean([m.get('regret_scaled', m['regret']) for m in all_metrics])
    avg_entropy = np.mean([m['entropy'] for m in all_metrics])
    avg_dist_bonus = np.mean([m.get('distance_bonus', 0) for m in all_metrics])
    overlap_rate = np.mean([m.get('has_overlap', False) for m in all_metrics]) * 100
    
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Greedy Length: {avg_greedy:.2f}")
    print(f"Average Regret: {avg_regret:.4f}")
    print(f"Average Regret (scaled): {avg_regret_scaled:.4f}")
    print(f"Average Entropy: {avg_entropy:.3f}")
    print(f"Average Distance Bonus: {avg_dist_bonus:.4f}")
    print(f"Overlap Rate: {overlap_rate:.1f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Diffusion V2 Generator")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--num-nodes", type=int, default=20,
                        help="Number of nodes")
    parser.add_argument("--map-size", type=int, default=30,
                        help="Map size")
    parser.add_argument("--total-demand", type=int, default=60,
                        help="Total demand")
    parser.add_argument("--max-c", type=int, default=10,
                        help="Max demand per node")
    parser.add_argument("--capacity", type=int, default=30,
                        help="Vehicle capacity")
    parser.add_argument("--depot-x", type=int, default=15,
                        help="Depot X coordinate")
    parser.add_argument("--depot-y", type=int, default=15,
                        help="Depot Y coordinate")
    parser.add_argument("--ddim-steps", type=int, default=10,
                        help="DDIM sampling steps")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save visualization")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_and_visualize(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        map_size=args.map_size,
        total_demand=args.total_demand,
        max_c=args.max_c,
        capacity=args.capacity,
        depot=(args.depot_x, args.depot_y),
        ddim_steps=args.ddim_steps,
        save_path=args.save_path,
        device=args.device,
    )
