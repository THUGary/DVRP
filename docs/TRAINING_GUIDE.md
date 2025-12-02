# DVRP Training Guide

本文档说明如何使用 shell 脚本训练静态和动态 VRP 模型。

## 概述

系统使用两阶段训练：
1. **静态模型 (Static VRP)**: POMO 风格的模型，在静态 VRP 问题上训练
2. **动态适配器 (Dynamic Adapter)**: 在静态模型基础上训练的适配层，用于动态 VRP

## 固定参数 (DO NOT CHANGE)

以下参数在整个系统中**固定不变**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `DEMAND_NORM` | 30 | 需求归一化常数 = 车辆容量 |
| `DEFAULT_CAPACITY` | 30 | 车辆容量 (模型看到 30/30 = 1.0) |
| `DEFAULT_MAX_DEMAND` | 5 | 每个节点最大需求 (模型看到 5/30 ≈ 0.167) |

**为什么固定这些参数？**
- 模型在归一化空间 [0,1] 中训练，车辆容量 = 1.0
- `DEMAND_NORM = 30` 意味着真实容量 30 对应模型中的 1.0
- 每个节点需求 1-5，归一化后为 0.033-0.167
- 一辆车最多服务约 6 个满载节点

## 可变参数

以下参数可以根据需要调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PROBLEM_SIZE` / `TOTAL_DEMAND` | 20 | 客户节点数量 (需求点数量) |
| `MAP_SIZE` / `GRID_SIZE` | 20 | 地图大小 (COORD_NORM) |
| `NUM_AGENTS` | 2 | 车辆数量 |

---

## 训练静态模型

### 基本用法

```bash
# 使用默认参数训练 (20 节点, 20x20 地图)
bash scripts/train_static.sh

# 训练 50 节点问题
PROBLEM_SIZE=50 bash scripts/train_static.sh

# 训练 50 节点, 40x40 地图
PROBLEM_SIZE=50 MAP_SIZE=40 bash scripts/train_static.sh

# 更多车辆
NUM_AGENTS=3 PROBLEM_SIZE=30 bash scripts/train_static.sh
```

### 训练参数调整

```bash
# 更多 epochs (更好效果，更长时间)
EPOCHS=1000 bash scripts/train_static.sh

# 更大 POMO size (更好效果，更多内存)
POMO_SIZE=200 bash scripts/train_static.sh

# 更大 batch size (更快训练)
BATCH_SIZE=128 bash scripts/train_static.sh

# 组合使用
EPOCHS=1000 POMO_SIZE=100 BATCH_SIZE=64 PROBLEM_SIZE=50 bash scripts/train_static.sh
```

### 恢复训练

```bash
# 从检查点恢复
RESUME_FROM=checkpoints/static_vrp_v2/checkpoint_n20_ep100.pt bash scripts/train_static.sh
```

### 输出文件

训练后检查点保存在 `checkpoints/static_vrp_v2/`:
- `best_n{PROBLEM_SIZE}.pt` - 最佳模型
- `checkpoint_n{PROBLEM_SIZE}_ep{N}.pt` - 周期检查点

---

## 训练动态适配器

### 前提条件

需要先有对应的静态模型检查点。

### 基本用法

```bash
# 使用默认参数训练
bash scripts/train_dynamic.sh

# 训练 50 节点版本
NUM_DEMANDS=50 bash scripts/train_dynamic.sh

# 训练 50 节点, 40x40 地图
NUM_DEMANDS=50 GRID_SIZE=40 bash scripts/train_dynamic.sh
```

### 指定静态检查点

```bash
# 使用特定静态模型
STATIC_CKPT=checkpoints/static_vrp_v2/best_n50.pt NUM_DEMANDS=50 bash scripts/train_dynamic.sh
```

### 训练参数

```bash
# 更多 epochs
EPOCHS=100 bash scripts/train_dynamic.sh

# 使用负载均衡训练
USE_BALANCE_TRAINING=true BALANCE_WEIGHT=0.5 bash scripts/train_dynamic.sh
```

### 输出文件

训练后检查点保存在 `checkpoints/dynamic_adapter_v2/`:
- `best_adapter_rl.pt` - 最佳适配器
- `adapter_rl_ep{N}.pt` - 周期检查点

---

## 评估模型

### 基本用法

```bash
# 评估默认模型
bash scripts/evaluate_distributions.sh

# 评估 50 节点模型
TOTAL_DEMAND=50 MODEL_CHECKPOINTS=checkpoints/static_vrp_v2/best_n50.pt bash scripts/evaluate_distributions.sh

# 在 40x40 地图上评估
TOTAL_DEMAND=50 MAP_WIDTH=40 MAP_HEIGHT=40 bash scripts/evaluate_distributions.sh
```

### 比较多个模型

```bash
# 比较多个检查点
MODEL_CHECKPOINTS="best_n20=checkpoints/static_vrp_v2/best_n20.pt,best_n50=checkpoints/static_vrp_v2/best_n50.pt" bash scripts/evaluate_distributions.sh
```

---

## 运行单次演示

```bash
# 使用静态规划器
bash scripts/run_dvrp.sh

# 使用规则规划器
PLANNER=rule bash scripts/run_dvrp.sh

# 禁用渲染 (测试)
RENDER=false bash scripts/run_dvrp.sh
```

---

## 完整训练流程示例

### 示例 1: 训练 20 节点问题

```bash
# Step 1: 训练静态模型
PROBLEM_SIZE=20 EPOCHS=500 bash scripts/train_static.sh

# Step 2: 训练动态适配器
NUM_DEMANDS=20 bash scripts/train_dynamic.sh

# Step 3: 评估
TOTAL_DEMAND=20 bash scripts/evaluate_distributions.sh
```

### 示例 2: 训练 50 节点, 40x40 地图

```bash
# Step 1: 训练静态模型
PROBLEM_SIZE=50 MAP_SIZE=40 EPOCHS=1000 bash scripts/train_static.sh

# Step 2: 训练动态适配器
NUM_DEMANDS=50 GRID_SIZE=40 STATIC_CKPT=checkpoints/static_vrp_v2/best_n50.pt bash scripts/train_dynamic.sh

# Step 3: 评估
TOTAL_DEMAND=50 MAP_WIDTH=40 MAP_HEIGHT=40 \
  MODEL_CHECKPOINTS=checkpoints/static_vrp_v2/best_n50.pt \
  bash scripts/evaluate_distributions.sh
```

### 示例 3: 多配置比较实验

```bash
# 训练不同规模的模型
for size in 20 30 50; do
    PROBLEM_SIZE=$size EPOCHS=500 SAVE_DIR=checkpoints/static_vrp_v2_exp bash scripts/train_static.sh
done

# 评估所有模型
MODEL_CHECKPOINTS="n20=checkpoints/static_vrp_v2_exp/best_n20.pt,n30=checkpoints/static_vrp_v2_exp/best_n30.pt,n50=checkpoints/static_vrp_v2_exp/best_n50.pt" \
  bash scripts/evaluate_distributions.sh
```

---

## 归一化方案说明

### 坐标归一化
- 原始坐标: `[0, MAP_SIZE]`
- 归一化: `coord / COORD_NORM`
- 模型空间: `[0, 1]`

### 需求归一化
- 原始需求: `[1, 5]`
- 归一化: `demand / DEMAND_NORM = demand / 30`
- 模型空间: `[0.033, 0.167]`

### 车辆容量归一化
- 原始容量: `30`
- 归一化: `capacity / DEMAND_NORM = 30 / 30 = 1.0`
- 模型空间: `1.0`

### 重要提示

⚠️ **DEMAND_NORM = 30 是固定的，不要修改！**

修改 `DEMAND_NORM` 会导致：
- 需要重新训练所有模型
- 旧模型检查点不兼容

✅ **可以安全修改的参数：**
- `COORD_NORM` (地图大小) - 需要确保 MAP_SIZE = COORD_NORM
- `PROBLEM_SIZE` (节点数量) - 每个规模需要单独的模型
- `NUM_AGENTS` (车辆数量)

---

## 常见问题

### Q: 模型评估时失败率很高？
A: 确保使用匹配的模型检查点。例如，`best_n20.pt` 只能用于 20 节点问题。

### Q: 如何支持更大的地图？
A: 调整 `MAP_SIZE`/`GRID_SIZE`，同时确保 `configs.py` 中的 `COORD_NORM` 与之匹配。

### Q: 为什么容量固定为 30？
A: 这是设计选择。`DEMAND_NORM = 30` 使得 `capacity/DEMAND_NORM = 1.0`，简化了模型的归一化。

### Q: 如何删除旧的检查点？
A: 可以安全删除 `checkpoints/` 下的旧文件：
```bash
rm -rf checkpoints/static_vrp_v2_new/  # 旧的训练结果
rm -rf checkpoints/dynamic_adapter_v2_new/  # 旧的适配器
```

---

## 配置文件参考

关键配置在 `configs.py`:

```python
COORD_NORM: float = 20.0     # 地图大小 (可变)
DEMAND_NORM: float = 30.0    # 车辆容量 (固定)
DEFAULT_CAPACITY: int = 30   # 车辆容量 (固定)
DEFAULT_MAX_DEMAND: int = 5  # 最大需求 (固定)
```
