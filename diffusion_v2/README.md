# Diffusion V2: VRP 需求生成器

全新的 VRP 需求生成 Diffusion 架构，使用 PPO 进行对抗训练。

**📖 后处理（Postprocessing）**: `sampler.generate()` 已集成完整后处理流程，返回可直接用于求解的 VRP 实例数据。关键点简洁说明如下：

- 流程（按顺序）:
  1. DDIM 采样 → 归一化坐标与需求比率
  2. 物理斥力修正（physics_unrolling）→ 推开过近点
  3. 坐标/需求缩放到整数（demand_coord_scale）
  4. 需求裁剪与重分配（demand_clip）
  5. 合并重复节点（merge_duplicate_nodes）→ 将完全相同坐标的节点合并，需求累加并 clip 到 capacity

```
原始模型输出
    ↓
1. DDIM 采样 (sample_ddim)
   - 输出: 归一化坐标 [0,1] + 需求比率 [0,1]
    ↓
2. 物理斥力去重叠 (physics_unrolling)
   - 检测并推开距离过近的节点
   - 输出: 修正后的归一化坐标
    ↓
3. 坐标/需求缩放 (demand_coord_scale)
   - 坐标 → 整数 [0, map_size-1]
   - 需求比率 → 整数 [1, total_demand]
    ↓
4. 需求裁剪 (demand_clip)
   - 限制单点需求 ≤ max_c
   - 溢出需求重新分配
    ↓
5. 合并重复节点 (merge_duplicate_nodes) ✨ 新增
   - 合并坐标完全相同的节点
   - 需求累加（不超过 max_capacity）
   - 返回合并比率作为监控指标
    ↓
最终可用数据
```

- 返回值说明（`coords, demands, info`）:
  - `coords`: (Batch, M, 2) 整数坐标（M <= N，含 padding）
  - `demands`: (Batch, M, 1) 整数需求（padding 节点为 0）
  - `info`: 包含 `retries`, `has_overlap`, `demand_clip_success`, `overlap_ratios`, `original_num_nodes`, `merged_num_nodes`

- 使用建议:
  - 在训练 / cotrain / run_dvrp 中均直接使用 `sampler.generate(..., merge_duplicates=True)`（默认即 True），输出可以直接传入 planner 或环境。
  - 若需调试原始（未合并）数据，可传 `merge_duplicates=False`。

（后处理细节已直接并入本 README，无需查看单独文件）

## 架构特点

### 1. VRPDiffusionPolicy (Nano-DiT)
- **3层 Transformer Encoder** (禁止 U-Net)
- **无位置编码**: 保持点集排列不变性
- **线性层变换**: 非 patch embedding
- **AdaLN 注入**: 时间步 + 全局条件
- **解耦输出头**:
  - `CoordHead`: 输出 (N, 2)，Sigmoid 归一化到 [0,1]
  - `DemandHead`: 输出 (N, 1)，Softmax 表示需求比例

### 2. InferenceSampler（含完整后处理）
- `sample_ddim`: 10步快速 DDIM 采样
- `physics_unrolling`: 粒子斥力后处理，解决节点重叠
- `demand_coord_scale`: 坐标/需求缩放到整数
- `demand_clip`: 需求裁剪与重分配
- `merge_duplicate_nodes`: ✨ **去重合并**（坐标相同的节点）

**重要**: `sampler.generate()` 集成了所有后处理步骤，返回**可直接使用**的 VRP 实例数据。

### 3. VRPGeneratorEnv
- `GreedyPlanner`: 贪心规划器
- 奖励计算:
  - `Valid_Mask`: 重叠惩罚
  - `Regret`: (Greedy路径长度 - baseline) / baseline
  - `Entropy`: 空间分布熵
  - 最终奖励 = Valid_Mask × (Regret + λ × Entropy)

### 4. PPOAgent
- Diffusion 模型作为 Actor
- 简化版 PPO Clip Loss
- 奖励归一化

## 模型超参数

所有超参数在 `model.py` 中定义为静态常量:

```python
# Nano-DiT 架构参数
HIDDEN_DIM = 256          # Transformer 隐藏维度
NUM_HEADS = 4             # 注意力头数
NUM_LAYERS = 3            # Transformer 层数
MLP_RATIO = 4.0           # MLP 扩展比例

# Diffusion 参数
NUM_DIFFUSION_STEPS = 1000  # 扩散步数
BETA_START = 1e-4           # beta 起始值
BETA_END = 0.02             # beta 结束值

# 输入输出维度
NODE_INPUT_DIM = 3          # (x, y, demand_logit)
COORD_OUTPUT_DIM = 2        # (x, y)
DEMAND_OUTPUT_DIM = 1       # (demand_ratio)
GLOBAL_COND_DIM = 3         # (depot_x, depot_y, target_load_ratio)
```

## 使用方法

### 训练

```bash
# 使用默认配置训练
bash scripts/train_diffusion_v2.sh

# 或直接运行
python3 -m diffusion_v2.train \
    --epochs 1000 \
    --episodes-per-epoch 32 \
    --num-nodes 20 \
    --map-size 30 \
    --total-demand 60 \
    --max-c 10
```

### 可视化

```bash
# 可视化随机模型
bash scripts/visualize_diffusion_v2.sh

# 可视化训练好的模型
bash scripts/visualize_diffusion_v2.sh checkpoints/diffusion_v2/run_xxx/best.pth
```

### 代码集成

```python
from diffusion_v2 import DiffusionV2Generator

# 加载训练好的模型
generator = DiffusionV2Generator.load("checkpoints/diffusion_v2/best.pth")

# 生成 VRP 实例
demands = generator.generate(
    num_nodes=20,
    map_size=30,
    total_demand=60,
    max_c=10,
)
# demands: List[(x, y, demand), ...]

# 或生成环境格式 (带时间信息)
env_demands = generator.generate_for_env(
    num_nodes=20,
    map_size=30,
    total_demand=60,
    max_c=10,
)
# env_demands: List[(x, y, t_arrival, demand, t_due), ...]
```

## 文件结构

```
diffusion_v2/
├── __init__.py       # 模块导出
├── model.py          # VRPDiffusionPolicy (Nano-DiT)
├── sampler.py        # InferenceSampler (DDIM + 后处理)
├── env.py            # VRPGeneratorEnv (Greedy + 奖励)
├── ppo.py            # PPOAgent (PPO 训练)
├── train.py          # 主训练循环
├── adapter.py        # 兼容旧接口的适配器
├── visualize.py      # 可视化工具
└── README.md         # 本文档
```

## 输入输出规范

### VRPDiffusionPolicy

**输入**:
- `noisy_state`: (Batch, N, 3) - 加噪节点状态 [x, y, demand_logit]
- `timestep`: (Batch,) - 扩散时间步
- `global_condition`: (Batch, 3) - [depot_x, depot_y, target_load_ratio]

**输出**:
- `pred_coords`: (Batch, N, 2) - 预测坐标 [0, 1]
- `pred_demand_ratios`: (Batch, N, 1) - 需求比例分布

### InferenceSampler.generate

**输入**:
- `global_condition`: (Batch, 3)
- `num_nodes`: int
- `map_size`: int
- `total_demand`: int
- `max_c`: int

**输出**:
- `final_coords`: (Batch, N, 2) - 整数坐标
- `final_demands`: (Batch, N, 1) - 整数需求

### VRPGeneratorEnv.get_reward

**输入**:
- `coords`: (N, 2) - 整数坐标
- `demands`: (N,) - 整数需求
- `depot`: Tuple[int, int]
- `has_overlap`: bool

**输出**:
- `reward`: float
- `metrics`: Dict[str, float]

## 与旧模型的区别

| 特性 | 旧 DemandDiffusionModel | 新 VRPDiffusionPolicy |
|------|------------------------|----------------------|
| 架构 | MLP | Nano-DiT (Transformer) |
| 输出 | 联合 (N, 5) | 解耦: Coord + Demand |
| 位置编码 | 有 | 无 (排列不变性) |
| 条件注入 | 直接拼接 | AdaLN |
| 后处理 | 无 | 物理斥力 + 需求裁剪 |
| 训练方式 | REINFORCE | PPO |
| 时间维度 | 支持 | 静态 VRP |

## 注意事项

1. **静态 VRP**: 本模型专注于静态 VRP，不处理时间维度 (arrival_time, due_time)
2. **物理后处理**: 自动解决节点重叠问题
3. **需求约束**: 自动满足 total_demand 和 max_c 约束
4. **兼容接口**: 通过 `DiffusionV2Generator` 适配器与旧代码兼容
