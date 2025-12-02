# DVRP Training and Testing Guide

本文档包含完整的训练和测试流程脚本。

## 环境准备

```bash
# 激活 conda 环境
source /home/user0/anaconda3/etc/profile.d/conda.sh
conda activate dvrp
cd /home/user0/DVRP-11.23
```

---

## 1. 静态模型训练 (Static POMO Model)

### 1.1 基础训练（从头开始）

```bash
python -m training_v2.train_static \
    --problem-size 20 \
    --pomo-size 20 \
    --epochs 50 \
    --episodes-per-epoch 2000 \
    --batch-size 64 \
    --lr 1e-4 \
    --save-dir checkpoints/static_vrp_v2 \
    --device cuda
```
### 1.2 继续训练（从检查点恢复）

```bash
python -m training_v2.train_static \
    --problem-size 20 \
    --pomo-size 20 \
    --epochs 50 \
    --episodes-per-epoch 2000 \
    --batch-size 64 \
    --lr 1e-4 \
    --save-dir checkpoints/static_vrp_v2 \
    --resume checkpoints/static_vrp_v2/best_n20.pt \
    --device cuda
```

### 1.3 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--problem-size` | VRP 问题规模（节点数） | 20 |
| `--pomo-size` | POMO 采样数量 | 20 |
| `--epochs` | 训练轮数 | 50 |
| `--episodes-per-epoch` | 每轮训练的 episode 数 | 2000 |
| `--batch-size` | 批次大小 | 64 |
| `--lr` | 学习率 | 1e-4 |
| `--save-dir` | 检查点保存目录 | checkpoints/static_vrp_v2 |
| `--resume` | 恢复训练的检查点路径 | None |
| `--device` | 训练设备 | cuda |

### 1.4 使用 shell 脚本

```bash
# 使用脚本训练静态模型
PROBLEM_SIZE=50 MAP_SIZE=40 bash scripts/train_static.sh

# 指定 POMO 大小和 epochs
PROBLEM_SIZE=30 EPOCHS=100 POMO_SIZE=100 bash scripts/train_static.sh
```

> 脚本会自动打印当前配置（地图大小、容量、需求数量），通过环境变量覆盖即可实现不同组合。

---

## 2. 动态适配器训练 (Dynamic Adapter)

### 2.1 基础 RL 训练

```bash
python -m training_v2.train_dynamic \
    --static-checkpoint checkpoints/static_vrp_v2/best_n20.pt \
    --mode rl \
    --num-agents 2 \
    --num-demands 20 \
    --epochs 30 \
    --episodes-per-epoch 50 \
    --save-dir checkpoints/dynamic_adapter_v2 \
    --device cuda
```

### 2.2 带负载均衡的 RL 训练（推荐）

```bash
python -m training_v2.train_dynamic \
    --static-checkpoint checkpoints/static_vrp_v2/best_n20.pt \
    --mode rl \
    --num-agents 2 \
    --num-demands 20 \
    --epochs 30 \
    --episodes-per-epoch 50 \
    --use-balance-training \
    --balance-weight 0.5 \
    --save-dir checkpoints/dynamic_balanced_v2 \
    --device cuda
```

### 2.3 监督学习训练

```bash
python -m training_v2.train_dynamic \
    --static-checkpoint checkpoints/static_vrp_v2/best_n20.pt \
    --mode supervised \
    --num-agents 2 \
    --num-demands 20 \
    --epochs 30 \
    --episodes-per-epoch 50 \
    --save-dir checkpoints/dynamic_supervised_v2 \
    --device cuda
```

### 2.4 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--static-checkpoint` | 静态模型检查点路径 | 必需 |
| `--mode` | 训练模式 (rl/supervised) | rl |
| `--num-agents` | 车辆数量 | 2 |
| `--num-demands` | 需求点数量 | 20 |
| `--epochs` | 训练轮数 | 30 |
| `--episodes-per-epoch` | 每轮 episode 数 | 50 |
| `--use-balance-training` | 启用负载均衡训练 | False |
| `--no-balance-training` | 禁用负载均衡训练 | - |
| `--balance-weight` | 均衡损失权重 | 0.5 |
| `--save-dir` | 检查点保存目录 | checkpoints/dynamic_adapter_v2 |
| `--device` | 训练设备 | cuda |

### 2.5 使用 shell 脚本

```bash
# 使用脚本训练动态适配器
NUM_DEMANDS=30 GRID_SIZE=30 bash scripts/train_dynamic.sh

# 指定静态模型并启用负载均衡
STATIC_CKPT=checkpoints/static_vrp_v2/best_n30.pt USE_BALANCE_TRAINING=true \
    BALANCE_WEIGHT=0.5 \
    bash scripts/train_dynamic.sh
```

> 脚本会打印固定参数（capacity=30、max_c=5）和当前可变参数，通过环境变量快速尝试不同地图/需求组合。

---

## 3. 评估脚本

### 3.1 跨分布评估（推荐）

比较规则方法和模型方法在不同需求分布下的表现：

```bash
# 评估规则方法 vs 动态模型
python evaluate_distributions.py \
    --rule-based greedy \
    --model-checkpoints checkpoints/dynamic_balanced_v2/best_adapter_rl.pt \
    --num-agents 2 \
    --num-runs 10 \
    --out-dir outputs/eval_results

# 评估多个模型
python evaluate_distributions.py \
    --rule-based greedy optimize \
    --model-checkpoints \
        "baseline=checkpoints/dynamic_adapter_v2/best_adapter.pt" \
        "balanced=checkpoints/dynamic_balanced_v2/best_adapter_rl.pt" \
    --num-agents 2 \
    --num-runs 20 \
    --out-dir outputs/eval_comparison
```

### 3.2 静态需求评估

```bash
python evaluate_distributions.py \
    --rule-based greedy \
    --model-checkpoints checkpoints/static_vrp_v2/best_n20.pt \
    --static-demands \
    --num-agents 2 \
    --num-runs 10 \
    --out-dir outputs/eval_static
```

### 3.3 单次运行可视化

```bash
# 使用规则方法运行（带渲染）
python run.py \
    --planner rule \
    --rule-mode greedy \
    --render \
    --num-agents 2

# 使用静态模型运行
python run.py \
    --planner static \
    --static-ckpt checkpoints/static_vrp_v2/best_n20.pt \
    --render \
    --num-agents 2

# 使用动态模型运行
python run.py \
    --planner dynamic \
    --static-ckpt checkpoints/static_vrp_v2/best_n20.pt \
    --adapter-ckpt checkpoints/dynamic_balanced_v2/best_adapter_rl.pt \
    --render \
    --num-agents 2

# 静态需求场景
python run.py \
    --planner static \
    --static-ckpt checkpoints/static_vrp_v2/best_n20.pt \
    --static-demands \
    --total-demand 200 \
    --render
```

### 3.4 评估参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rule-based` | 规则方法 (greedy/optimize) | None |
| `--model-checkpoints` | 模型检查点列表 | [] |
| `--num-agents` | 车辆数量 | 2 |
| `--num-runs` | 每个分布的运行次数 | 10 |
| `--static-demands` | 使用静态需求 | False |
| `--out-dir` | 输出目录 | outputs/eval |
| `--plot-metrics` | 绘图指标 | service_ratio,total_distance |
| `--use-hungarian` | 使用匈牙利算法分配 | False |

---

## 4. 完整训练流程

### 4.1 一键训练脚本

创建 `train_all.sh`:

```bash
#!/bin/bash

# 激活环境
source /home/user0/anaconda3/etc/profile.d/conda.sh
conda activate dvrp
cd /home/user0/DVRP-11.23

echo "=========================================="
echo "Step 1: Training Static Model"
echo "=========================================="

python -m training_v2.train_static \
    --problem-size 20 \
    --pomo-size 20 \
    --epochs 50 \
    --episodes-per-epoch 2000 \
    --batch-size 64 \
    --lr 1e-4 \
    --save-dir checkpoints/static_vrp_v2 \
    --device cuda

echo "=========================================="
echo "Step 2: Training Dynamic Adapter (with Balance)"
echo "=========================================="

python -m training_v2.train_dynamic \
    --static-checkpoint checkpoints/static_vrp_v2/best_n20.pt \
    --mode rl \
    --num-agents 2 \
    --num-demands 20 \
    --epochs 50 \
    --episodes-per-epoch 100 \
    --use-balance-training \
    --balance-weight 0.5 \
    --save-dir checkpoints/dynamic_balanced_v2 \
    --device cuda

echo "=========================================="
echo "Step 3: Evaluation"
echo "=========================================="

python evaluate_distributions.py \
    --rule-based greedy \
    --model-checkpoints checkpoints/dynamic_balanced_v2/best_adapter_rl.pt \
    --num-agents 2 \
    --num-runs 20 \
    --out-dir outputs/final_eval

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
```

运行：
```bash
chmod +x train_all.sh
./train_all.sh
```

---

## 5. 检查点目录结构

```
checkpoints/
├── static_vrp_v2/
│   ├── best_n20.pt          # 最佳静态模型
│   └── checkpoint_n20.pt    # 最新检查点
├── dynamic_adapter_v2/
│   ├── best_adapter.pt      # 最佳适配器（基础）
│   └── checkpoint_adapter.pt
├── dynamic_balanced_v2/
│   ├── best_adapter_rl.pt   # 最佳适配器（带均衡）
│   └── checkpoint_adapter_rl.pt
└── dynamic_supervised_v2/
    └── best_adapter_supervised.pt
```

---

## 6. 输出说明

### 6.1 训练输出

训练过程会输出：
- **Reward**: 累积奖励（越高越好）
- **Loss**: 策略损失
- **Balance Score**: 负载均衡分数（0-1，越高越均衡）
- **DistCV**: 距离变异系数（越低越均衡）

示例输出：
```
Training Dynamic VRP Model on cuda
Mode: rl
Balance Training: True (weight=0.5)
Epoch 1/30: Reward=12.40, Loss=1.43, Balance=0.878, DistCV=0.154
  New best: 12.40
Epoch 9/30: Reward=26.31, Loss=0.84, Balance=0.902, DistCV=0.113
  New best: 26.31
```

### 6.2 评估输出

评估结果包括：
- **service_ratio**: 服务率（已服务需求/总需求）
- **total_distance**: 总行驶距离
- **failure_flag**: 失败标志（静态需求时）
- **episode_steps**: 回合步数

输出目录包含：
```
outputs/eval_results/eval_YYYYMMDD-HHMMSS/
├── service_ratio_by_distribution.png
├── total_distance_by_distribution.png
└── episode_length_by_distribution.png  # (静态需求时)
```

---

## 7. 常见问题

### Q1: CUDA 内存不足
```bash
# 减小批次大小
--batch-size 32

# 或使用 CPU
--device cpu
```

### Q2: 训练不收敛
```bash
# 降低学习率
--lr 5e-5

# 增加训练数据
--episodes-per-epoch 5000
```

### Q3: 负载不均衡
```bash
# 增加均衡权重
--balance-weight 1.0

# 或使用匈牙利算法
--use-hungarian
```

---

## 8. 性能基准

| 方法 | Service Ratio | Total Distance | 训练时间 |
|------|--------------|----------------|----------|
| Rule (Greedy) | ~97% | ~105 | N/A |
| Static POMO | ~90% | ~85 | ~2h |
| Dynamic Adapter | ~60% | ~70 | ~30min |
| Dynamic + Balance | ~65% | ~70 | ~35min |

*注：以上为 2 车辆、20 需求点的参考值*

---

## 9. 多车辆段分配 (Multi-Vehicle Segment Distribution)

### 9.1 工作原理

静态模型和动态模型都使用相同的多车辆路径分配策略：

1. **生成完整路径**: 模型生成一条访问所有节点的完整路径
2. **在depot处切分**: 当车辆容量耗尽时返回depot，形成多个段（segment）
3. **均衡分配**: 使用工作量平衡策略将段分配给不同车辆

```
完整路径: depot → 1 → 2 → depot → 3 → 4 → 5 → depot
    ↓
段切分:   [1, 2] 和 [3, 4, 5]
    ↓
分配给车辆:
  - 车辆 0: [1, 2]
  - 车辆 1: [3, 4, 5]
```

### 9.2 分配策略

| 策略 | 说明 | 使用场景 |
|------|------|----------|
| `sequential` | 按顺序分配段 | 简单场景 |
| `round_robin` | 轮流分配 | 段数量较多 |
| `balanced` | 按工作量平衡 | **推荐** |
| `distance_aware` | 考虑车辆位置 | 车辆分散时 |

### 9.3 测试段分配

```bash
# 运行段分配测试
python test_segment_distribution.py

# 评估训练模型的段分配效果
python evaluate_segment_distribution.py
```

### 9.4 代码使用

```python
from models_v2.static_model import StaticVRPModel

model = StaticVRPModel()
model.load_state_dict(torch.load("checkpoints/static_vrp_v2/best_n20.pt")['model_state_dict'])

# solve() 方法自动使用段分配
distances, routes = model.solve(
    depot_xy, node_xy, node_demand,
    pomo_size=20,
    num_vehicles=2,  # 指定车辆数量
)

# routes[0] = [vehicle_0_nodes, vehicle_1_nodes]
print(f"车辆 0 路径: {routes[0][0]}")
print(f"车辆 1 路径: {routes[0][1]}")
```

### 9.5 V2Planner 使用

```python
from agent.planner.v2_planner import V2Planner

# 静态模式 - 一次性规划所有节点
planner = V2Planner(mode="static")
result = planner.plan(observations, agent_states, depot, t, horizon)

# 动态模式 - 也使用相同的段分配策略
planner = V2Planner(mode="dynamic")
result = planner.plan(observations, agent_states, depot, t, horizon)

# result = [agent_0_targets, agent_1_targets]
```

### 9.6 平衡性指标

评估多车辆分配的平衡性：

```python
from models_v2.segment_distributor import get_distribution_balance

# vehicle_segments: 每个车辆分配到的段
balance = get_distribution_balance(vehicle_segments)

print(f"距离变异系数 (CV): {balance['cv_distance']:.3f}")  # 越小越平衡
print(f"需求变异系数 (CV): {balance['cv_demand']:.3f}")
```

理想的平衡比例 (min/max) 应该接近 1.0。

---

## 10. 扩展训练

### 10.1 更大规模问题

```bash
# 50 节点问题
python -m training_v2.train_static \
    --problem-size 50 \
    --pomo-size 50 \
    --epochs 100 \
    --batch-size 32 \
    --save-dir checkpoints/static_n50
```

### 10.2 多车辆训练

```bash
# 4 车辆
python -m training_v2.train_dynamic \
    --static-checkpoint checkpoints/static_vrp_v2/best_n20.pt \
    --num-agents 4 \
    --use-balance-training \
    --balance-weight 0.8 \
    --save-dir checkpoints/dynamic_4agents
```

### 10.3 自定义分布训练

支持的分布类型：
- `uniform`: 均匀分布
- `gaussian`: 高斯分布
- `cluster`: 聚类分布
- `explosion`: 爆发分布
- `implosion`: 内聚分布
