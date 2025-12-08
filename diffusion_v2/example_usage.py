#!/usr/bin/env python3
"""
示例：在 Co-training 或其他场景中使用 Diffusion V2 生成器

展示如何正确使用 sampler.generate() 获得可直接使用的 VRP 实例数据
"""

import torch
from diffusion_v2.model import VRPDiffusionPolicy, create_global_condition
from diffusion_v2.sampler import InferenceSampler


def generate_vrp_instance(
    model_path: str,
    depot: tuple,
    num_nodes: int = 20,
    map_size: int = 30,
    total_demand: int = 60,
    max_c: int = 10,
    capacity: int = 30,
    device: str = "auto",
) -> dict:
    """
    生成单个 VRP 实例
    
    Args:
        model_path: 模型检查点路径
        depot: depot 坐标 (x, y)
        num_nodes: 节点数
        map_size: 地图大小
        total_demand: 总需求量
        max_c: 最大单点需求
        capacity: 车辆容量
        device: 设备 (auto/cuda/cpu)
        
    Returns:
        instance: {
            "coords": (N, 2) numpy array, 整数坐标
            "demands": (N,) numpy array, 整数需求
            "depot": (x, y) depot 坐标
            "capacity": 车辆容量
            "map_size": 地图大小
            "info": 生成信息（含合并比率等）
        }
    """
    # 设置设备
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # 加载模型
    model = VRPDiffusionPolicy().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 创建采样器
    sampler = InferenceSampler(model, ddim_steps=10)
    
    # 创建条件
    depot_norm_x = depot[0] / (map_size - 1)
    depot_norm_y = depot[1] / (map_size - 1)
    
    global_cond = create_global_condition(
        depot_x=depot_norm_x,
        depot_y=depot_norm_y,
        total_demand=total_demand,
        capacity=capacity,
        batch_size=1,
        device=device,
    )
    
    # 生成（包含完整后处理）
    coords, demands, info = sampler.generate(
        global_condition=global_cond,
        num_nodes=num_nodes,
        map_size=map_size,
        total_demand=total_demand,
        max_c=max_c,
        batch_size=1,
        apply_physics=True,      # 物理斥力去重叠
        merge_duplicates=True,   # 合并重复节点
    )
    
    # 转换为 numpy（去掉 batch 维度和 padding）
    coords = coords.squeeze(0).cpu().numpy()  # (N, 2)
    demands = demands.squeeze(0).squeeze(-1).cpu().numpy()  # (N,)
    
    # 过滤掉 padding 节点（需求为 0）
    valid_mask = demands > 0
    coords = coords[valid_mask]
    demands = demands[valid_mask]
    
    return {
        "coords": coords,
        "demands": demands,
        "depot": depot,
        "capacity": capacity,
        "map_size": map_size,
        "info": info,
    }


def main():
    """示例使用"""
    print("=== Diffusion V2 生成器使用示例 ===\n")
    
    # 配置
    model_path = "checkpoints/diffusion_v2/best.pth"  # 修改为你的模型路径
    depot = (15, 15)
    
    # 生成
    try:
        instance = generate_vrp_instance(
            model_path=model_path,
            depot=depot,
            num_nodes=20,
            map_size=30,
            total_demand=60,
            max_c=10,
            capacity=30,
        )
        
        print("✓ 生成成功！")
        print(f"\n实例信息:")
        print(f"  节点数: {len(instance['coords'])}")
        print(f"  总需求: {instance['demands'].sum()}")
        print(f"  Depot: {instance['depot']}")
        print(f"  容量: {instance['capacity']}")
        
        print(f"\n生成统计:")
        info = instance['info']
        print(f"  原始节点数: {info['original_num_nodes']}")
        print(f"  合并后节点数: {info['merged_num_nodes']}")
        print(f"  合并比率: {info['overlap_ratios'][0]:.1%}")
        print(f"  重试次数: {info['retries']}")
        
        print(f"\n前5个节点:")
        for i in range(min(5, len(instance['coords']))):
            x, y = instance['coords'][i]
            d = instance['demands'][i]
            print(f"  Node {i}: 坐标=({x}, {y}), 需求={d}")
        
        print(f"\n✓ 数据可直接用于 VRP 求解！")
        
    except FileNotFoundError:
        print(f"✗ 模型文件未找到: {model_path}")
        print(f"请先训练模型或修改 model_path")
        print(f"\n训练命令: bash scripts/train_diffusion_v2.sh")


if __name__ == "__main__":
    main()
