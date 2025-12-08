#!/usr/bin/env python3
"""
测试修复：曼哈顿距离、padding 过滤、新 baseline 公式
"""

import torch
import numpy as np
from diffusion_v2.env import VRPGeneratorEnv, manhattan_distance

def test_manhattan_distance():
    """测试曼哈顿距离计算"""
    print("=== 测试 1: 曼哈顿距离 ===")
    a = np.array([0, 0], dtype=np.float32)
    b = np.array([3, 4], dtype=np.float32)
    manhattan = manhattan_distance(a, b)
    euclidean = np.linalg.norm(b - a)
    print(f"点 {a} 到 {b}:")
    print(f"  曼哈顿距离: {manhattan} (应为 7.0)")
    print(f"  欧氏距离: {euclidean:.2f} (5.0)")
    assert manhattan == 7.0, "曼哈顿距离计算错误"
    print("✓ 通过\n")


def test_padding_filter():
    """测试 padding 节点过滤"""
    print("=== 测试 2: 过滤 padding 节点 ===")
    env = VRPGeneratorEnv(map_size=30, capacity=30)
    
    # 创建包含 padding 的实例
    coords = torch.tensor([[5, 5], [10, 10], [15, 15], [0, 0], [0, 0]])
    demands = torch.tensor([5, 5, 5, 0, 0])  # 后两个是 padding
    depot = (15, 15)
    
    reward, metrics = env.get_reward(coords, demands, depot)
    print(f"输入: 5 个节点（3 个有效 + 2 个 padding）")
    print(f"  原始节点数: {metrics['original_num_nodes']}")
    print(f"  合并后节点数: {metrics['merged_num_nodes']}")
    print(f"  Greedy 长度: {metrics['greedy_length']:.2f}")
    print(f"  Baseline 长度: {metrics['baseline_length']:.2f}")
    print(f"  Regret: {metrics['regret']:.4f}")
    print(f"  Greedy >= Baseline: {metrics['greedy_length'] >= metrics['baseline_length']}")
    
    # 验证：Greedy 应该 >= Baseline（因为 baseline 是下界）
    assert metrics['greedy_length'] >= metrics['baseline_length'], \
        f"Greedy ({metrics['greedy_length']}) 应该 >= Baseline ({metrics['baseline_length']})"
    print("✓ 通过\n")


def test_new_baseline():
    """测试新 baseline 公式"""
    print("=== 测试 3: 新 Baseline 公式 ===")
    env = VRPGeneratorEnv(map_size=30, capacity=30)
    
    # 简单实例：3个节点，depot在中心
    coords = torch.tensor([[0, 0], [30, 0], [0, 30]])
    demands = torch.tensor([20, 20, 20])  # 总需求60，容量30，需要2车次
    depot = (15, 15)
    
    reward, metrics = env.get_reward(coords, demands, depot)
    
    # 手动计算预期值
    # 节点到 depot 的曼哈顿距离: (15+15), (15+15), (15+15) = 30, 30, 30
    # 平均距离 = 30
    # 最少车次 = ceil(60/30) = 2
    # Baseline = 2 * 2 * 30 = 120
    expected_baseline = 120.0
    
    print(f"3节点实例（总需求60，容量30）:")
    print(f"  最少车次: 2")
    print(f"  到depot曼哈顿距离: [30, 30, 30]")
    print(f"  平均距离: 30.0")
    print(f"  预期baseline: {expected_baseline:.2f}")
    print(f"  实际baseline: {metrics['baseline_length']:.2f}")
    print(f"  Greedy: {metrics['greedy_length']:.2f}")
    print(f"  Regret: {metrics['regret']:.4f}")
    
    assert abs(metrics['baseline_length'] - expected_baseline) < 1.0, \
        f"Baseline ({metrics['baseline_length']}) 与预期 ({expected_baseline}) 差异过大"
    assert metrics['regret'] >= 0, "Regret 应该非负（Greedy >= Baseline）"
    print("✓ 通过\n")


def test_random_instances():
    """测试随机实例的统计特性"""
    print("=== 测试 4: 随机实例统计 ===")
    env = VRPGeneratorEnv(map_size=30, capacity=30)
    
    num_tests = 20
    regrets = []
    
    for i in range(num_tests):
        coords = torch.randint(0, 30, (20, 2))
        demands = torch.randint(1, 10, (20,))
        depot = (15, 15)
        
        reward, metrics = env.get_reward(coords, demands, depot)
        regrets.append(metrics['regret'])
        
        # 验证 Greedy >= Baseline
        if metrics['greedy_length'] < metrics['baseline_length']:
            print(f"  [警告] 实例 {i}: Greedy={metrics['greedy_length']:.2f} < Baseline={metrics['baseline_length']:.2f}")
    
    regrets = np.array(regrets)
    print(f"随机生成 {num_tests} 个实例:")
    print(f"  Regret 范围: [{regrets.min():.4f}, {regrets.max():.4f}]")
    print(f"  Regret 均值: {regrets.mean():.4f}")
    print(f"  Regret 为正的比例: {(regrets >= 0).mean():.1%}")
    
    # 大部分 regret 应该为正
    positive_ratio = (regrets >= 0).mean()
    assert positive_ratio >= 0.8, f"Regret 为正的比例 ({positive_ratio:.1%}) 应该 >= 80%"
    print("✓ 通过\n")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("测试修复：曼哈顿距离 + padding 过滤 + 新 baseline")
    print("=" * 60)
    print()
    
    try:
        test_manhattan_distance()
        test_padding_filter()
        test_new_baseline()
        test_random_instances()
        
        print("=" * 60)
        print("✓✓✓ 所有测试通过！")
        print("=" * 60)
        print()
        print("主要修复:")
        print("1. ✓ 距离度量改为曼哈顿距离（车辆只能上下左右移动）")
        print("2. ✓ 过滤 padding 节点（demand=0）不参与计算")
        print("3. ✓ Baseline 改为下界估计：2 * min_trips * avg_depot_dist")
        print()
        print("预期训练效果:")
        print("- Regret 应该大部分为正值（Greedy >= Baseline）")
        print("- Reward 会更合理（不再出现大量负值）")
        print("- 训练日志现在显示 Baseline 长度，便于监控")
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
