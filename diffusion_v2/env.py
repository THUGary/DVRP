"""
VRPGeneratorEnv: VRP 生成器环境与奖励计算

功能:
- GreedyPlanner: 贪心规划器
- Valid_Mask: 重叠惩罚
- Regret: (Greedy路径长度 - baseline) / baseline
- Entropy: 空间分布熵
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# ==============================================================================
# 环境超参数 (静态常量)
# ==============================================================================

# 奖励权重 (调优: Regret 主导，Entropy 正则化)
# 
# 设计原则:
# - Regret 范围约 [-0.05, +0.15]，缩放后应为主要信号 (目标: [-0.5, +1.5])
# - Entropy 范围约 [0.7, 0.9]，作为正则化避免退化 (目标贡献: ~0.05-0.1)
# - Overlap 惩罚应允许部分梯度流过，但明确指示重叠是不好的
#
REGRET_SCALE: float = 10.0            # Regret 缩放系数 (增大: 让 regret 成为主导)
ENTROPY_LAMBDA: float = 0.1           # 熵奖励权重 (降低: 只作为正则化，避免四角收敛)
OVERLAP_PENALTY: float = -1.0         # 重叠惩罚 (进一步降低: 保留更多梯度信号)
DISTANCE_BONUS_SCALE: float = 0.05    # 距离奖励缩放 (降低: 减少干扰)

# Greedy Planner 参数
DEFAULT_CAPACITY: int = 30            # 默认车辆容量

# 熵计算参数
ENTROPY_GRID_SIZE: int = 5            # 熵计算网格划分数

# 重叠检测参数 (整数坐标空间)
MIN_INT_DISTANCE: float = 1.0         # 最小整数距离阈值 (< 1 表示同一格子)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算曼哈顿距离（L1 距离）
    
    在环境中，车辆只能上下左右移动，不能斜向移动
    
    Args:
        a: 点A坐标
        b: 点B坐标
        
    Returns:
        distance: 曼哈顿距离
    """
    return np.abs(a - b).sum()


def merge_overlapping_nodes(
    coords: torch.Tensor,
    demands: torch.Tensor,
    max_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    合并重叠节点 (坐标相同的节点)
    
    策略:
    - 将坐标完全相同的节点合并为一个
    - 合并后的需求 = min(sum_of_demands, max_capacity)
    - 返回合并比率作为监控指标
    
    Args:
        coords: (N, 2) 整数坐标
        demands: (N,) 整数需求
        max_capacity: 最大容量 (需求上限)
        
    Returns:
        merged_coords: (M, 2) 合并后坐标 (M <= N)
        merged_demands: (M,) 合并后需求
        overlap_ratio: 重叠比率 = (N - M) / N
    """
    device = coords.device
    coords_np = coords.cpu().numpy()
    demands_np = demands.cpu().numpy()
    
    n = len(coords_np)
    if n == 0:
        return coords, demands, 0.0
    
    # 用字典按坐标分组
    coord_to_demands: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        key = (int(coords_np[i, 0]), int(coords_np[i, 1]))
        if key not in coord_to_demands:
            coord_to_demands[key] = []
        coord_to_demands[key].append(int(demands_np[i]))
    
    # 构建合并后的坐标和需求
    merged_coords_list = []
    merged_demands_list = []
    
    for (x, y), demand_list in coord_to_demands.items():
        merged_coords_list.append([x, y])
        # 合并需求，不超过容量上限
        merged_demand = min(sum(demand_list), max_capacity)
        merged_demands_list.append(merged_demand)
    
    m = len(merged_coords_list)
    overlap_ratio = (n - m) / n if n > 0 else 0.0
    
    # 转回 tensor
    merged_coords = torch.tensor(merged_coords_list, dtype=coords.dtype, device=device)
    merged_demands = torch.tensor(merged_demands_list, dtype=demands.dtype, device=device)
    
    return merged_coords, merged_demands, overlap_ratio


@dataclass
class VRPInstance:
    """VRP 问题实例"""
    coords: torch.Tensor      # (N, 2) 节点坐标 (整数)
    demands: torch.Tensor     # (N,) 节点需求 (整数)
    depot: Tuple[int, int]    # depot 坐标
    capacity: int             # 车辆容量
    map_size: int             # 地图大小


class GreedyPlanner:
    """
    贪心 VRP 规划器
    
    策略: 每次选择最近的可服务节点
    """
    
    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = capacity
    
    def solve(self, instance: VRPInstance) -> Tuple[float, List[List[int]]]:
        """
        求解 VRP 实例
        
        Args:
            instance: VRP 问题实例
            
        Returns:
            total_distance: 总行驶距离
            routes: 每辆车的路径 (节点索引列表)
        """
        coords = instance.coords.cpu().numpy()
        demands = instance.demands.cpu().numpy()
        depot = np.array(instance.depot, dtype=np.float32)
        capacity = instance.capacity
        
        # 过滤掉需求为 0 的节点（padding 节点）
        valid_mask = demands > 0
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return 0.0, []
        
        valid_coords = coords[valid_mask]
        valid_demands = demands[valid_mask]
        
        num_nodes = len(valid_coords)
        visited = np.zeros(num_nodes, dtype=bool)
        routes = []
        total_distance = 0.0
        
        while not visited.all():
            # 开始新路径
            route = []
            current_pos = depot.copy()
            remaining_capacity = capacity
            
            while True:
                # 找到最近的可服务节点（使用曼哈顿距离）
                best_idx = -1
                best_dist = float('inf')
                
                for i in range(num_nodes):
                    if visited[i]:
                        continue
                    if valid_demands[i] > remaining_capacity:
                        continue
                    
                    dist = manhattan_distance(valid_coords[i], current_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                
                if best_idx == -1:
                    # 没有可服务节点，返回 depot
                    break
                
                # 服务该节点
                route.append(int(valid_indices[best_idx]))  # 返回原始索引
                visited[best_idx] = True
                total_distance += best_dist
                current_pos = valid_coords[best_idx].copy()
                remaining_capacity -= valid_demands[best_idx]
            
            # 返回 depot（使用曼哈顿距离）
            total_distance += manhattan_distance(current_pos, depot)
            
            if route:
                routes.append(route)
        
        return total_distance, routes


def compute_mst_length(coords: np.ndarray) -> float:
    """
    计算最小生成树长度 (Prim 算法，使用曼哈顿距离)
    
    Args:
        coords: (N, 2) 节点坐标
        
    Returns:
        mst_length: MST 长度
    """
    n = len(coords)
    if n <= 1:
        return 0.0
    
    # 计算距离矩阵（使用曼哈顿距离）
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = manhattan_distance(coords[i], coords[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # Prim 算法
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, float('inf'))
    min_dist[0] = 0
    mst_length = 0.0
    
    for _ in range(n):
        # 找到最小距离的节点
        u = -1
        min_d = float('inf')
        for i in range(n):
            if not in_mst[i] and min_dist[i] < min_d:
                min_d = min_dist[i]
                u = i
        
        if u == -1:
            break
        
        in_mst[u] = True
        mst_length += min_d
        
        # 更新距离
        for v in range(n):
            if not in_mst[v] and dist_matrix[u, v] < min_dist[v]:
                min_dist[v] = dist_matrix[u, v]
    
    return mst_length


def compute_spatial_entropy(coords: torch.Tensor, grid_size: int = ENTROPY_GRID_SIZE) -> float:
    """
    计算空间分布熵
    
    将地图划分为 grid_size x grid_size 的网格，计算节点分布的熵
    
    Args:
        coords: (N, 2) 归一化坐标 [0, 1]
        grid_size: 网格划分数
        
    Returns:
        entropy: 空间分布熵
    """
    coords_np = coords.cpu().numpy()
    n = len(coords_np)
    
    if n == 0:
        return 0.0
    
    # 将坐标映射到网格
    grid_indices = (coords_np * grid_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, grid_size - 1)
    
    # 统计每个网格的节点数
    counts = np.zeros((grid_size, grid_size))
    for idx in grid_indices:
        counts[idx[0], idx[1]] += 1
    
    # 计算概率分布
    probs = counts.flatten() / n
    probs = probs[probs > 0]  # 去除零概率
    
    # 计算熵
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # 归一化到 [0, 1] (最大熵 = log(grid_size^2))
    max_entropy = np.log(grid_size * grid_size)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return float(normalized_entropy)


class VRPGeneratorEnv:
    """
    VRP 生成器环境
    
    用于评估生成的 VRP 实例质量
    
    奖励 = Valid_Mask * (Regret_Scaled + λ * Entropy + Distance_Bonus)
    
    其中:
    - Valid_Mask: 1 if 无重叠, 乘以 OVERLAP_PENALTY 如果有重叠
    - Regret: (greedy_length - baseline_length) / baseline_length * REGRET_SCALE
    - Entropy: 空间分布熵 (归一化到 [0, 1])
    - Distance_Bonus: 节点远离 depot 的平均距离奖励 (鼓励挑战性分布)
    - Baseline: MST长度 + 2/capacity * Σ(depot到节点的距离)
    """
    
    def __init__(
        self,
        map_size: int,
        capacity: int = DEFAULT_CAPACITY,
        entropy_lambda: float = ENTROPY_LAMBDA,
        regret_scale: float = REGRET_SCALE,
        overlap_penalty: float = OVERLAP_PENALTY,
        distance_bonus_scale: float = DISTANCE_BONUS_SCALE,
    ):
        self.map_size = map_size
        self.capacity = capacity
        self.entropy_lambda = entropy_lambda
        self.regret_scale = regret_scale
        self.overlap_penalty = overlap_penalty
        self.distance_bonus_scale = distance_bonus_scale
        self.planner = GreedyPlanner(capacity)
    
    def compute_baseline(self, instance: VRPInstance) -> float:
        """
        计算 baseline 长度（最下限估计）
        
        Baseline = 2 * min_trips * avg_depot_distance
        
        这是总距离的下界：每辆车至少要从 depot 出发并返回 depot
        
        Args:
            instance: VRP 实例
            
        Returns:
            baseline: baseline 长度
        """
        coords = instance.coords.cpu().numpy().astype(np.float32)
        depot = np.array(instance.depot, dtype=np.float32)
        demands = instance.demands.cpu().numpy()
        
        # 过滤掉需求为 0 的节点（padding 节点）
        valid_mask = demands > 0
        if not valid_mask.any():
            return 1e-6
        
        valid_coords = coords[valid_mask]
        valid_demands = demands[valid_mask]
        
        # 总需求量 / 容量 = 最少需要的车次
        total_demand = valid_demands.sum()
        min_trips = max(1, int(np.ceil(total_demand / instance.capacity)))
        
        # depot 到各有效节点的平均距离（使用曼哈顿距离）
        avg_depot_dist = np.mean([manhattan_distance(c, depot) for c in valid_coords])
        
        # Baseline = 2 * 最少车次 * 平均到depot距离
        # 每辆车至少要走：depot → 某节点 → depot，即 2 * avg_depot_dist
        baseline = 2 * min_trips * avg_depot_dist
        
        return max(baseline, 1e-6)  # 避免除零
    
    def get_reward(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        depot: Tuple[int, int],
        has_overlap: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励
        
        新策略: 先合并重叠节点，再计算奖励
        - 不再对重叠进行惩罚
        - 合并后的实例用于计算 Greedy 和 Baseline
        - overlap_ratio 仅作为监控指标
        
        Args:
            coords: (N, 2) 整数坐标
            demands: (N, 1) 或 (N,) 整数需求
            depot: depot 坐标
            has_overlap: 是否有重叠 (来自 sampler，现在仅作参考)
            
        Returns:
            reward: 标量奖励
            metrics: 包含详细指标的字典
        """
        # 处理需求维度
        if demands.dim() == 2:
            demands = demands.squeeze(-1)
        
        # === 合并重叠节点 ===
        merged_coords, merged_demands, overlap_ratio = merge_overlapping_nodes(
            coords, demands, self.capacity
        )
        original_num_nodes = coords.shape[0]
        merged_num_nodes = merged_coords.shape[0]
        
        # 创建合并后的 VRP 实例
        instance = VRPInstance(
            coords=merged_coords,
            demands=merged_demands,
            depot=depot,
            capacity=self.capacity,
            map_size=self.map_size,
        )
        
        # 计算 Greedy 解 (使用合并后的实例)
        greedy_length, routes = self.planner.solve(instance)
        
        # 计算 Baseline (使用合并后的实例)
        baseline_length = self.compute_baseline(instance)
        
        # 计算 Regret (越大越好，说明 greedy 效果越差 = 问题越难)
        regret = (greedy_length - baseline_length) / baseline_length
        regret_scaled = regret * self.regret_scale
        
        # 计算空间熵 (使用合并后的坐标)
        normalized_coords = merged_coords.float() / (self.map_size - 1)
        entropy = compute_spatial_entropy(normalized_coords)
        
        # 计算距离奖励 (使用合并后的坐标，仅计算有效节点)
        coords_np = merged_coords.cpu().numpy()
        demands_np = merged_demands.cpu().numpy()
        depot_np = np.array(depot, dtype=np.float32)
        
        # 过滤掉需求为 0 的节点
        valid_mask = demands_np > 0
        if valid_mask.any():
            valid_coords = coords_np[valid_mask]
            avg_depot_dist = np.mean([manhattan_distance(c, depot_np) for c in valid_coords])
            max_possible_dist = 2 * self.map_size  # 曼哈顿距离最大值（从一角到对角）
            normalized_dist = avg_depot_dist / max_possible_dist
            distance_bonus = normalized_dist * self.distance_bonus_scale
        else:
            distance_bonus = 0.0
        
        # 计算总奖励 (不再有重叠惩罚)
        reward = regret_scaled + self.entropy_lambda * entropy + distance_bonus
        
        metrics = {
            "greedy_length": greedy_length,
            "baseline_length": baseline_length,
            "regret": regret,
            "regret_scaled": regret_scaled,
            "entropy": entropy,
            "distance_bonus": distance_bonus,
            "valid_mask": 1.0,  # 现在总是有效
            "reward": reward,
            "num_routes": len(routes),
            "has_overlap": overlap_ratio > 0,  # 兼容旧接口
            "overlap_ratio": overlap_ratio,     # 新增: 重叠比率
            "original_num_nodes": original_num_nodes,
            "merged_num_nodes": merged_num_nodes,
        }
        
        return reward, metrics
    
    def get_batch_reward(
        self,
        coords_batch: torch.Tensor,
        demands_batch: torch.Tensor,
        depot: Tuple[int, int],
        overlap_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        批量计算奖励
        
        Args:
            coords_batch: (Batch, N, 2) 整数坐标
            demands_batch: (Batch, N, 1) 整数需求
            depot: depot 坐标
            overlap_mask: (Batch,) 是否有重叠 (仅作参考，不再影响奖励)
            
        Returns:
            rewards: (Batch,) 奖励
            metrics: 包含详细指标的字典
        """
        batch_size = coords_batch.shape[0]
        device = coords_batch.device
        
        rewards = []
        all_metrics = {
            "greedy_length": [],
            "baseline_length": [],
            "regret": [],
            "entropy": [],
            "valid_mask": [],
            "num_routes": [],
            "overlap_ratio": [],  # 新增: 重叠比率
        }
        
        for b in range(batch_size):
            coords = coords_batch[b]
            demands = demands_batch[b]
            has_overlap = overlap_mask[b].item() if overlap_mask is not None else False
            
            reward, metrics = self.get_reward(coords, demands, depot, has_overlap)
            rewards.append(reward)
            
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k].append(v)
        
        # 转换为张量
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        for k in all_metrics:
            all_metrics[k] = torch.tensor(all_metrics[k], device=device)
        
        return rewards, all_metrics


if __name__ == "__main__":
    # 测试环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VRPGeneratorEnv on {device}")
    
    # 创建环境
    env = VRPGeneratorEnv(map_size=30, capacity=30)
    
    # 创建测试实例
    num_nodes = 20
    coords = torch.randint(0, 30, (num_nodes, 2), device=device)
    demands = torch.randint(1, 10, (num_nodes,), device=device)
    depot = (15, 15)
    
    # 计算奖励
    reward, metrics = env.get_reward(coords, demands, depot, has_overlap=False)
    
    print(f"\nSingle instance results:")
    print(f"  Coords shape: {coords.shape}")
    print(f"  Demands sum: {demands.sum().item()}")
    print(f"  Reward: {reward:.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 测试批量
    batch_size = 4
    coords_batch = torch.randint(0, 30, (batch_size, num_nodes, 2), device=device)
    demands_batch = torch.randint(1, 10, (batch_size, num_nodes, 1), device=device)
    overlap_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    overlap_mask[0] = True  # 第一个样本有重叠
    
    rewards, batch_metrics = env.get_batch_reward(
        coords_batch, demands_batch, depot, overlap_mask
    )
    
    print(f"\nBatch results:")
    print(f"  Rewards: {rewards}")
    print(f"  Mean regret: {batch_metrics['regret'].mean():.4f}")
    print(f"  Mean entropy: {batch_metrics['entropy'].mean():.4f}")
    
    # 测试 MST
    test_coords = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]
    ], dtype=np.float32)
    mst_len = compute_mst_length(test_coords)
    print(f"\nMST test (square): {mst_len:.4f} (expected ~3.0)")
    
    print("\n✓ VRPGeneratorEnv test passed!")
