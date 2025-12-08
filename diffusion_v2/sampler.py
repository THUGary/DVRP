"""
InferenceSampler: DDIM 采样器与后处理

功能:
- sample_ddim: 10步快速 DDIM 采样
- physics_unrolling: 粒子斥力后处理 (解决重叠)
- demand_coord_scale: 坐标/需求缩放
- demand_clip: 需求裁剪与重分配
- merge_duplicate_nodes: 去重合并 (坐标相同的节点)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import VRPDiffusionPolicy

# ==============================================================================
# 采样器超参数 (静态常量)
# ==============================================================================

# DDIM 采样参数
DEFAULT_DDIM_STEPS: int = 10          # 默认 DDIM 采样步数
DEFAULT_DDIM_ETA: float = 0.0         # DDIM 随机性 (0=确定性)

# 物理斥力参数
REPULSION_ITERATIONS: int = 200       # 斥力迭代次数
REPULSION_STRENGTH: float = 0.2       # 斥力强度
MIN_DISTANCE_THRESHOLD: float = 0.03  # 最小距离阈值 (归一化坐标) - 降低以适应更多点
REPULSION_DECAY: float = 0.995        # 斥力衰减 (更慢衰减)

# 需求处理参数
MAX_REDISTRIBUTE_ITERATIONS: int = 100  # 最大重分配迭代次数


class InferenceSampler:
    """
    VRP Diffusion 推理采样器
    
    功能:
    1. DDIM 快速采样
    2. 物理斥力后处理 (解决重叠)
    3. 坐标/需求缩放与裁剪
    """
    
    def __init__(
        self,
        model: "VRPDiffusionPolicy",
        ddim_steps: int = DEFAULT_DDIM_STEPS,
        ddim_eta: float = DEFAULT_DDIM_ETA,
    ):
        """
        Args:
            model: VRPDiffusionPolicy 模型
            ddim_steps: DDIM 采样步数
            ddim_eta: DDIM 随机性参数
        """
        self.model = model
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.device = next(model.parameters()).device
        
    @torch.no_grad()
    def sample_ddim(
        self,
        global_condition: torch.Tensor,
        num_nodes: int,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM 快速采样
        
        Args:
            global_condition: (Batch, 3) 全局条件
            num_nodes: 节点数量
            batch_size: batch 大小
            
        Returns:
            pred_coords: (Batch, N, 2) 预测坐标 [0, 1]
            pred_demand_ratios: (Batch, N, 1) 需求比例
        """
        self.model.eval()
        
        # 获取 diffusion 参数
        num_train_steps = self.model.betas.shape[0]
        alphas_cumprod = self.model.alphas_cumprod
        
        # 创建 DDIM 时间步序列
        step_ratio = num_train_steps // self.ddim_steps
        timesteps = torch.arange(0, num_train_steps, step_ratio, device=self.device).flip(0)
        
        # 初始化纯噪声
        x_t = torch.randn(batch_size, num_nodes, 3, device=self.device)
        
        for i, t_curr in enumerate(timesteps):
            t_batch = t_curr.expand(batch_size)
            
            # 模型预测
            pred_coords, pred_demand_ratios = self.model(x_t, t_batch, global_condition)
            
            # 重建 x_0 预测
            # pred_x0 = (x, y, demand_logit) 格式
            # 我们用模型的输出构建 x_0
            pred_x0 = torch.cat([
                pred_coords,  # (B, N, 2)
                pred_demand_ratios,  # (B, N, 1)
            ], dim=-1)  # (B, N, 3)
            
            # 计算当前和下一步的 alpha
            alpha_t = alphas_cumprod[t_curr]
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t_next = alphas_cumprod[t_next]
            else:
                alpha_t_next = torch.tensor(1.0, device=self.device)
            
            # DDIM 更新公式
            # x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * direction
            
            # 计算 direction (指向 x_t 的方向)
            sigma_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)
            ) if self.ddim_eta > 0 and i < len(timesteps) - 1 else 0.0
            
            # 预测噪声方向
            pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)
            
            # 计算下一步
            dir_xt = torch.sqrt(1 - alpha_t_next - sigma_t ** 2) * pred_noise
            x_t = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt
            
            # 添加随机噪声 (如果 eta > 0)
            if sigma_t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise
        
        # 最终输出
        final_coords = torch.sigmoid(x_t[..., :2])  # (B, N, 2)
        final_demand_logits = x_t[..., 2:3]  # (B, N, 1)
        final_demand_ratios = F.softmax(final_demand_logits.squeeze(-1), dim=-1).unsqueeze(-1)
        
        return final_coords, final_demand_ratios
    
    @staticmethod
    def physics_unrolling(
        coords: torch.Tensor,
        min_distance: float = MIN_DISTANCE_THRESHOLD,
        iterations: int = REPULSION_ITERATIONS,
        strength: float = REPULSION_STRENGTH,
        decay: float = REPULSION_DECAY,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        粒子斥力后处理: 将重叠的点推开
        
        使用两阶段策略:
        1. 初始扩散: 检测高度聚集区域，对聚集点添加随机偏移
        2. 斥力精调: 迭代应用斥力，微调点位置
        
        Args:
            coords: (Batch, N, 2) 坐标 [0, 1]
            min_distance: 最小距离阈值
            iterations: 迭代次数
            strength: 斥力强度
            decay: 斥力衰减
            
        Returns:
            fixed_coords: (Batch, N, 2) 修正后的坐标
            overlap_mask: (Batch,) 是否仍有重叠 (True=有重叠)
        """
        batch_size, num_nodes, _ = coords.shape
        device = coords.device
        
        # 复制以避免修改原始数据
        fixed_coords = coords.clone()
        
        # 创建排除自身的 mask
        eye_mask = torch.eye(num_nodes, device=device).bool().unsqueeze(0)
        
        # ==== 阶段1: 初始扩散 ====
        # 计算初始重叠程度
        diff = fixed_coords.unsqueeze(2) - fixed_coords.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        dist_masked = dist.masked_fill(eye_mask, float('inf'))
        
        # 统计每个点有多少个重叠邻居
        overlap_count_per_point = (dist_masked < min_distance * 2).sum(dim=-1)  # (B, N)
        
        # 如果大部分点都聚集在一起，先随机分散
        max_overlap = overlap_count_per_point.max().item()
        if max_overlap > num_nodes * 0.3:  # 超过30%的点聚在一起
            # 计算当前的质心
            centroid = fixed_coords.mean(dim=1, keepdim=True)  # (B, 1, 2)
            
            # 给每个点一个随机偏移，偏移量与其重叠程度成正比
            # 使用均匀分布在圆上的初始位置
            angles = torch.linspace(0, 2 * 3.14159 * (1 - 1/num_nodes), num_nodes, device=device)
            # 添加随机扰动
            angles = angles + torch.rand(num_nodes, device=device) * 0.5
            radius = 0.3 + torch.rand(num_nodes, device=device) * 0.15  # 半径 0.3-0.45
            
            # 生成圆形分布
            new_x = 0.5 + radius * torch.cos(angles)
            new_y = 0.5 + radius * torch.sin(angles)
            
            # 对高度重叠的点使用新位置
            spread_coords = torch.stack([new_x, new_y], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
            
            # 混合原始坐标和分散坐标 (根据重叠程度)
            mix_ratio = (overlap_count_per_point.float() / max(max_overlap, 1)).unsqueeze(-1)  # (B, N, 1)
            fixed_coords = fixed_coords * (1 - mix_ratio) + spread_coords * mix_ratio
            fixed_coords = fixed_coords.clamp(0, 1)
        
        # ==== 阶段2: 斥力精调 ====
        current_strength = strength
        
        for iter_idx in range(iterations):
            # 计算所有点对距离 (B, N, N)
            diff = fixed_coords.unsqueeze(2) - fixed_coords.unsqueeze(1)  # (B, N, N, 2)
            dist = torch.norm(diff, dim=-1)  # (B, N, N)
            
            dist_masked = dist.masked_fill(eye_mask, float('inf'))
            
            # 找到距离小于阈值的点对
            too_close = dist_masked < min_distance  # (B, N, N)
            
            if not too_close.any():
                break
            
            # 处理完全重叠的点 (距离为 0) - 添加随机扰动
            zero_dist_mask = (dist_masked < 1e-6) & too_close
            if zero_dist_mask.any():
                # 为完全重叠的点添加随机扰动方向
                random_direction = torch.randn_like(diff)  # (B, N, N, 2)
                random_direction = random_direction / (torch.norm(random_direction, dim=-1, keepdim=True) + 1e-8)
                # 只应用于距离为 0 的点
                diff = torch.where(
                    zero_dist_mask.unsqueeze(-1).expand_as(diff),
                    random_direction * min_distance,  # 给一个随机方向
                    diff
                )
                # 重新计算 dist
                dist = torch.norm(diff, dim=-1)
                dist_masked = dist.masked_fill(eye_mask, float('inf'))
            
            # 计算斥力方向 (从 j 指向 i)
            # 归一化方向向量
            direction = diff / (dist.unsqueeze(-1) + 1e-8)  # (B, N, N, 2)
            
            # 计算斥力大小 (距离越近，斥力越大)
            force_magnitude = (min_distance - dist_masked).clamp(min=0)  # (B, N, N)
            
            # 应用斥力
            forces = direction * force_magnitude.unsqueeze(-1) * too_close.unsqueeze(-1).float()
            total_force = forces.sum(dim=2)  # (B, N, 2)
            
            # 更新坐标
            fixed_coords = fixed_coords + current_strength * total_force
            
            # 确保坐标在 [0, 1] 范围内
            fixed_coords = fixed_coords.clamp(0, 1)
            
            # 衰减斥力
            current_strength *= decay
        
        # 检查是否仍有重叠
        final_diff = fixed_coords.unsqueeze(2) - fixed_coords.unsqueeze(1)
        final_dist = torch.norm(final_diff, dim=-1)
        final_dist = final_dist.masked_fill(eye_mask, float('inf'))
        overlap_mask = (final_dist < min_distance).any(dim=-1).any(dim=-1)  # (B,)
        
        return fixed_coords, overlap_mask
    
    @staticmethod
    def demand_coord_scale(
        coords: torch.Tensor,
        demand_ratios: torch.Tensor,
        map_size: int,
        total_demand: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放坐标和需求到整数值
        
        Args:
            coords: (Batch, N, 2) 归一化坐标 [0, 1]
            demand_ratios: (Batch, N, 1) 需求比例
            map_size: 地图大小
            total_demand: 总需求量
            
        Returns:
            int_coords: (Batch, N, 2) 整数坐标 [0, map_size-1]
            int_demands: (Batch, N, 1) 整数需求量
        """
        # 坐标缩放并取整
        scaled_coords = coords * (map_size - 1)
        int_coords = scaled_coords.round().long()
        int_coords = int_coords.clamp(0, map_size - 1)
        
        # 需求缩放
        # 使用最大余数法保证总和精确
        raw_demands = demand_ratios.squeeze(-1) * total_demand  # (B, N)
        floor_demands = raw_demands.floor()  # (B, N)
        remainders = raw_demands - floor_demands  # (B, N)
        
        # 计算还需要分配的量
        deficit = total_demand - floor_demands.sum(dim=-1, keepdim=True)  # (B, 1)
        
        # 按余数大小分配剩余需求
        # 获取余数排序索引
        _, indices = remainders.sort(dim=-1, descending=True)
        
        # 创建额外分配 mask
        batch_size, num_nodes = raw_demands.shape
        extra = torch.zeros_like(floor_demands)
        for b in range(batch_size):
            num_extra = int(deficit[b].item())
            if num_extra > 0:
                extra[b, indices[b, :num_extra]] = 1
        
        int_demands = (floor_demands + extra).long().unsqueeze(-1)  # (B, N, 1)
        
        return int_coords, int_demands
    
    @staticmethod
    def demand_clip(
        demands: torch.Tensor,
        max_c: int,
        max_iterations: int = MAX_REDISTRIBUTE_ITERATIONS,
    ) -> Tuple[torch.Tensor, bool]:
        """
        裁剪超过 max_c 的需求并重新分配
        
        将超过 max_c 的需求强制为 max_c，多出来的需求均匀分配给其他容量不到 max_c 的节点
        
        Args:
            demands: (Batch, N, 1) 整数需求量
            max_c: 最大需求量
            max_iterations: 最大迭代次数
            
        Returns:
            clipped_demands: (Batch, N, 1) 裁剪后的需求量
            success: 是否成功 (所有节点 <= max_c)
        """
        batch_size, num_nodes, _ = demands.shape
        clipped = demands.clone().float()
        
        for _ in range(max_iterations):
            # 找到超过 max_c 的节点
            overflow_mask = clipped.squeeze(-1) > max_c  # (B, N)
            if not overflow_mask.any():
                # 所有节点都在限制内
                return clipped.long(), True
            
            # 计算溢出量
            overflow = (clipped.squeeze(-1) - max_c).clamp(min=0)  # (B, N)
            total_overflow = overflow.sum(dim=-1, keepdim=True)  # (B, 1)
            
            # 将溢出节点裁剪到 max_c
            clipped = clipped.squeeze(-1).clamp(max=max_c).unsqueeze(-1)
            
            # 找到可以接收更多需求的节点
            can_receive = clipped.squeeze(-1) < max_c  # (B, N)
            num_receivers = can_receive.sum(dim=-1, keepdim=True).float()  # (B, 1)
            
            if (num_receivers == 0).any():
                # 没有节点可以接收，失败
                return clipped.long(), False
            
            # 均匀分配溢出量
            per_node_add = total_overflow / (num_receivers + 1e-8)  # (B, 1)
            add_amount = per_node_add * can_receive.float()  # (B, N)
            
            # 使用最大余数法确保整数分配
            floor_add = add_amount.floor()
            remainder = add_amount - floor_add
            
            # 计算还需分配的量
            still_need = (total_overflow - floor_add.sum(dim=-1, keepdim=True)).round()
            
            # 按余数排序分配剩余
            _, indices = remainder.sort(dim=-1, descending=True)
            extra = torch.zeros_like(floor_add)
            for b in range(batch_size):
                n = int(still_need[b].item())
                if n > 0:
                    # 只分配给可接收的节点
                    valid_indices = indices[b][can_receive[b, indices[b]]][:n]
                    if len(valid_indices) > 0:
                        extra[b, valid_indices] = 1
            
            clipped = clipped + (floor_add + extra).unsqueeze(-1)
        
        # 检查最终结果
        success = (clipped.squeeze(-1) <= max_c).all().item()
        return clipped.long(), success
    
    @staticmethod
    def merge_duplicate_nodes(
        coords: torch.Tensor,
        demands: torch.Tensor,
        max_capacity: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        合并重复节点（坐标完全相同的节点）
        
        策略:
        - 按 batch 处理，对每个样本分别合并
        - 将坐标完全相同的节点合并为一个
        - 合并后的需求 = min(sum_of_demands, max_capacity)
        - 返回合并比率作为监控指标
        
        Args:
            coords: (Batch, N, 2) 整数坐标
            demands: (Batch, N, 1) 或 (Batch, N) 整数需求
            max_capacity: 最大容量（需求上限）
            
        Returns:
            merged_coords: (Batch, M, 2) 合并后坐标（M <= N，每个样本 M 可能不同）
            merged_demands: (Batch, M, 1) 合并后需求
            overlap_ratios: List[float] 每个样本的合并比率
        """
        device = coords.device
        batch_size = coords.shape[0]
        
        # 处理需求维度
        if demands.dim() == 3:
            demands = demands.squeeze(-1)  # (Batch, N)
        
        merged_coords_list = []
        merged_demands_list = []
        overlap_ratios = []
        
        for b in range(batch_size):
            coords_b = coords[b].cpu().numpy()  # (N, 2)
            demands_b = demands[b].cpu().numpy()  # (N,)
            n = len(coords_b)
            
            if n == 0:
                merged_coords_list.append(torch.zeros((0, 2), dtype=coords.dtype, device=device))
                merged_demands_list.append(torch.zeros((0,), dtype=demands.dtype, device=device))
                overlap_ratios.append(0.0)
                continue
            
            # 用字典按坐标分组
            coord_to_demands: Dict[Tuple[int, int], List[int]] = {}
            for i in range(n):
                key = (int(coords_b[i, 0]), int(coords_b[i, 1]))
                if key not in coord_to_demands:
                    coord_to_demands[key] = []
                coord_to_demands[key].append(int(demands_b[i]))
            
            # 构建合并后的坐标和需求
            merged_coords_b = []
            merged_demands_b = []
            
            for (x, y), demand_list in coord_to_demands.items():
                merged_coords_b.append([x, y])
                # 合并需求，不超过容量上限
                merged_demand = min(sum(demand_list), max_capacity)
                merged_demands_b.append(merged_demand)
            
            m = len(merged_coords_b)
            overlap_ratio = (n - m) / n if n > 0 else 0.0
            overlap_ratios.append(overlap_ratio)
            
            # 转回 tensor
            merged_coords_list.append(
                torch.tensor(merged_coords_b, dtype=coords.dtype, device=device)
            )
            merged_demands_list.append(
                torch.tensor(merged_demands_b, dtype=demands.dtype, device=device)
            )
        
        # 找到最大节点数，用于 padding
        max_nodes = max(len(c) for c in merged_coords_list)
        
        # Padding 到相同长度
        padded_coords = []
        padded_demands = []
        
        for coords_b, demands_b in zip(merged_coords_list, merged_demands_list):
            n_b = len(coords_b)
            if n_b < max_nodes:
                # Padding: 坐标用 0，需求用 0
                pad_coords = torch.zeros((max_nodes - n_b, 2), dtype=coords.dtype, device=device)
                pad_demands = torch.zeros((max_nodes - n_b,), dtype=demands.dtype, device=device)
                coords_b = torch.cat([coords_b, pad_coords], dim=0)
                demands_b = torch.cat([demands_b, pad_demands], dim=0)
            padded_coords.append(coords_b)
            padded_demands.append(demands_b)
        
        # Stack
        merged_coords = torch.stack(padded_coords, dim=0)  # (Batch, M, 2)
        merged_demands = torch.stack(padded_demands, dim=0).unsqueeze(-1)  # (Batch, M, 1)
        
        return merged_coords, merged_demands, overlap_ratios
    
    def generate(
        self,
        global_condition: torch.Tensor,
        num_nodes: int,
        map_size: int,
        total_demand: int,
        max_c: int,
        batch_size: int = 1,
        apply_physics: bool = True,
        max_retries: int = 5,
        merge_duplicates: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        完整生成流程（包含所有后处理）
        
        流程:
        1. DDIM 采样
        2. 物理斥力去重叠
        3. 坐标/需求缩放到整数
        4. 需求 clip 到 max_c
        5. 合并重复节点（坐标相同的节点）
        
        Args:
            global_condition: (Batch, 3) 全局条件
            num_nodes: 节点数
            map_size: 地图大小
            total_demand: 总需求量
            max_c: 最大单点需求
            batch_size: batch 大小
            apply_physics: 是否应用物理斥力
            max_retries: 最大重试次数
            merge_duplicates: 是否合并重复节点（默认 True）
            
        Returns:
            final_coords: (Batch, M, 2) 整数坐标（M <= N）
            final_demands: (Batch, M, 1) 整数需求
            info: dict 包含生成信息
        """
        info = {
            "retries": 0,
            "has_overlap": False,
            "demand_clip_success": True,
            "overlap_ratios": [],
            "original_num_nodes": num_nodes,
            "merged_num_nodes": num_nodes,
        }
        
        for retry in range(max_retries):
            # 1. DDIM 采样
            coords, demand_ratios = self.sample_ddim(
                global_condition=global_condition,
                num_nodes=num_nodes,
                batch_size=batch_size,
            )
            
            # 2. 物理斥力后处理
            if apply_physics:
                coords, overlap_mask = self.physics_unrolling(coords)
                info["has_overlap"] = overlap_mask.any().item()
            
            # 3. 坐标/需求缩放
            int_coords, int_demands = self.demand_coord_scale(
                coords, demand_ratios, map_size, total_demand
            )
            
            # 4. 需求裁剪
            int_demands, clip_success = self.demand_clip(int_demands, max_c)
            info["demand_clip_success"] = clip_success
            
            if clip_success:
                # 5. 合并重复节点（新增）
                if merge_duplicates:
                    int_coords, int_demands, overlap_ratios = self.merge_duplicate_nodes(
                        int_coords, int_demands, max_c
                    )
                    info["overlap_ratios"] = overlap_ratios
                    info["merged_num_nodes"] = int_coords.shape[1]
                
                info["retries"] = retry
                return int_coords, int_demands, info
        
        # 重试用尽（仍然尝试合并）
        if merge_duplicates:
            int_coords, int_demands, overlap_ratios = self.merge_duplicate_nodes(
                int_coords, int_demands, max_c
            )
            info["overlap_ratios"] = overlap_ratios
            info["merged_num_nodes"] = int_coords.shape[1]
        
        info["retries"] = max_retries
        return int_coords, int_demands, info


if __name__ == "__main__":
    # 测试采样器
    from .model import VRPDiffusionPolicy, create_global_condition
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing InferenceSampler on {device}")
    
    # 创建模型和采样器
    model = VRPDiffusionPolicy().to(device)
    sampler = InferenceSampler(model, ddim_steps=10)
    
    # 测试参数
    batch_size = 2
    num_nodes = 20
    map_size = 30
    total_demand = 60
    max_c = 10
    
    # 创建条件
    global_cond = create_global_condition(
        depot_x=0.5, depot_y=0.5,
        total_demand=total_demand, capacity=30,
        batch_size=batch_size, device=device
    )
    
    # 完整生成
    coords, demands, info = sampler.generate(
        global_condition=global_cond,
        num_nodes=num_nodes,
        map_size=map_size,
        total_demand=total_demand,
        max_c=max_c,
        batch_size=batch_size,
    )
    
    print(f"\nGeneration results:")
    print(f"  Coords shape: {coords.shape}, dtype: {coords.dtype}")
    print(f"  Coords range: [{coords.min().item()}, {coords.max().item()}]")
    print(f"  Demands shape: {demands.shape}, dtype: {demands.dtype}")
    print(f"  Demands sum per batch: {demands.sum(dim=1).squeeze()}")
    print(f"  Demands max: {demands.max().item()}")
    print(f"  Info: {info}")
    
    # 测试物理斥力
    print("\nTesting physics_unrolling:")
    test_coords = torch.tensor([
        [[0.5, 0.5], [0.51, 0.51], [0.1, 0.1]],  # 前两个点重叠
    ], device=device)
    fixed, overlap = InferenceSampler.physics_unrolling(test_coords)
    print(f"  Before: {test_coords}")
    print(f"  After: {fixed}")
    print(f"  Still overlapping: {overlap}")
    
    print("\n✓ InferenceSampler test passed!")
