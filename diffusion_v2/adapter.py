"""
Diffusion V2 适配器

提供与现有系统兼容的接口，替代旧的 DemandDiffusionModel
"""

from __future__ import annotations
import torch
from typing import List, Tuple, Optional
from pathlib import Path

from .model import VRPDiffusionPolicy, create_global_condition
from .sampler import InferenceSampler


class DiffusionV2Generator:
    """
    Diffusion V2 生成器适配器
    
    提供与旧接口兼容的方法，可直接替换 NetDemandGenerator 或 DemandDiffusionModel
    
    用法:
        generator = DiffusionV2Generator.load("checkpoints/diffusion_v2/best.pth")
        demands = generator.generate(
            num_nodes=20,
            map_size=30,
            total_demand=60,
            max_c=10,
        )
    """
    
    def __init__(
        self,
        model: VRPDiffusionPolicy,
        device: torch.device,
        ddim_steps: int = 10,
    ):
        self.model = model
        self.device = device
        self.sampler = InferenceSampler(model, ddim_steps=ddim_steps)
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: str = "auto",
        ddim_steps: int = 10,
    ) -> "DiffusionV2Generator":
        """
        从检查点加载生成器
        
        Args:
            checkpoint_path: 检查点路径
            device: 设备 ("auto", "cuda", "cpu")
            ddim_steps: DDIM 采样步数
            
        Returns:
            generator: DiffusionV2Generator 实例
        """
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        model = VRPDiffusionPolicy().to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        return cls(model, device, ddim_steps)
    
    @classmethod
    def create_random(
        cls,
        device: str = "auto",
        ddim_steps: int = 10,
    ) -> "DiffusionV2Generator":
        """
        创建随机初始化的生成器 (用于测试)
        """
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        model = VRPDiffusionPolicy().to(device)
        return cls(model, device, ddim_steps)
    
    def generate(
        self,
        num_nodes: int,
        map_size: int,
        total_demand: int,
        max_c: int,
        capacity: int = 30,
        depot: Tuple[int, int] = None,
        batch_size: int = 1,
    ) -> List[Tuple[int, int, int]]:
        """
        生成 VRP 实例
        
        Args:
            num_nodes: 节点数
            map_size: 地图大小
            total_demand: 总需求量
            max_c: 最大单点需求
            capacity: 车辆容量 (用于条件)
            depot: depot 坐标，默认为中心
            batch_size: batch 大小
            
        Returns:
            demands: List of (x, y, demand) tuples
        """
        if depot is None:
            depot = (map_size // 2, map_size // 2)
        
        # 归一化 depot
        depot_norm_x = depot[0] / (map_size - 1)
        depot_norm_y = depot[1] / (map_size - 1)
        
        # 创建条件
        global_cond = create_global_condition(
            depot_x=depot_norm_x,
            depot_y=depot_norm_y,
            total_demand=total_demand,
            capacity=capacity,
            batch_size=batch_size,
            device=self.device,
        )
        
        # 生成
        coords, demands, info = self.sampler.generate(
            global_condition=global_cond,
            num_nodes=num_nodes,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            batch_size=batch_size,
        )
        
        # 转换为 list of tuples
        # 只返回第一个 batch
        coords_np = coords[0].cpu().numpy()
        demands_np = demands[0].cpu().numpy().flatten()
        
        result = []
        for i in range(num_nodes):
            x, y = int(coords_np[i, 0]), int(coords_np[i, 1])
            d = int(demands_np[i])
            if d > 0:  # 只返回有需求的节点
                result.append((x, y, d))
        
        return result
    
    def generate_tensor(
        self,
        num_nodes: int,
        map_size: int,
        total_demand: int,
        max_c: int,
        capacity: int = 30,
        depot: Tuple[int, int] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 VRP 实例 (张量格式)
        
        Returns:
            coords: (Batch, N, 2) 整数坐标
            demands: (Batch, N, 1) 整数需求
        """
        if depot is None:
            depot = (map_size // 2, map_size // 2)
        
        depot_norm_x = depot[0] / (map_size - 1)
        depot_norm_y = depot[1] / (map_size - 1)
        
        global_cond = create_global_condition(
            depot_x=depot_norm_x,
            depot_y=depot_norm_y,
            total_demand=total_demand,
            capacity=capacity,
            batch_size=batch_size,
            device=self.device,
        )
        
        coords, demands, _ = self.sampler.generate(
            global_condition=global_cond,
            num_nodes=num_nodes,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            batch_size=batch_size,
        )
        
        return coords, demands
    
    def generate_for_env(
        self,
        num_nodes: int,
        map_size: int,
        total_demand: int,
        max_c: int,
        max_time: int = 1000,
        min_lifetime: int = 50,
        max_lifetime: int = 200,
        capacity: int = 30,
        depot: Tuple[int, int] = None,
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        为环境生成需求 (包含时间信息的格式)
        
        返回格式: List of (x, y, t_arrival, demand, t_due)
        由于 V2 不处理时间维度，这里将所有节点设为静态节点 (t=0, t_due=max_time)
        
        Returns:
            demands: List of (x, y, t_arrival, demand, t_due)
        """
        # 生成基础需求
        base_demands = self.generate(
            num_nodes=num_nodes,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            capacity=capacity,
            depot=depot,
        )
        
        # 转换为环境格式 (静态节点)
        result = []
        for x, y, d in base_demands:
            result.append((x, y, 0, d, max_time))
        
        return result


# 为了兼容旧代码的导入
DemandGenerator = DiffusionV2Generator


if __name__ == "__main__":
    # 测试适配器
    print("Testing DiffusionV2Generator...")
    
    # 创建随机生成器
    generator = DiffusionV2Generator.create_random()
    
    # 测试生成
    demands = generator.generate(
        num_nodes=20,
        map_size=30,
        total_demand=60,
        max_c=10,
    )
    
    print(f"\nGenerated {len(demands)} demand nodes:")
    for x, y, d in demands[:5]:
        print(f"  ({x}, {y}): demand={d}")
    if len(demands) > 5:
        print(f"  ... and {len(demands) - 5} more")
    
    total = sum(d for _, _, d in demands)
    print(f"\nTotal demand: {total}")
    
    # 测试张量生成
    coords, demands_t = generator.generate_tensor(
        num_nodes=20,
        map_size=30,
        total_demand=60,
        max_c=10,
        batch_size=4,
    )
    print(f"\nTensor generation:")
    print(f"  Coords shape: {coords.shape}")
    print(f"  Demands shape: {demands_t.shape}")
    print(f"  Demands sum: {demands_t.sum(dim=(1,2))}")
    
    # 测试环境格式
    env_demands = generator.generate_for_env(
        num_nodes=20,
        map_size=30,
        total_demand=60,
        max_c=10,
    )
    print(f"\nEnvironment format ({len(env_demands)} nodes):")
    for x, y, t, d, t_due in env_demands[:3]:
        print(f"  ({x}, {y}): t={t}, demand={d}, due={t_due}")
    
    print("\n✓ DiffusionV2Generator test passed!")
