对抗式统一入口

python train_adversarial.py --episodes 50 --device cuda --planner greedy --randomize_depot
生成器：数据与训练

python generate_data.py
python normalize_data.py
python train_diffusion_generator.py
python train_rl_diffusion_generator.py --episodes 50 --planner greedy --device cuda
规划器：数据与训练

python data_gen.py --episodes 200 --planner greedy --map_wid 20 --agent_num 2 --out_dir data
python train_model.py --data_dir data --map_wid 20 --agent_num 2 --epochs 200 --device cuda
python train_rl_planner.py --episodes 200 --algo ppo --ckpt_init planner_dynamic_20_2_200.pt --save_best planner_rl_best.pt --device cuda
## DVRP-11.5 概览

在网格地图上研究动态车辆路径规划（DVRP），包含需求生成器、规划器与控制器三大模块。环境以离散时间推进，车辆具有容量约束；仅在仓库格允许“同格叠放”，其他位置避免碰撞并自动处理过期需求。

### 亮点
- 网格环境：`environment/env.py` 实现了需求生成/过期、容量在仓库补满、非仓库碰撞规避等机制。
- 网格环境：`environment/env.py` 实现了需求生成/过期、容量在仓库补满、非仓库碰撞规避等机制；`environment/env_tensor.py` 提供了等价的张量化批量环境，可在 GPU 上并行推进多局仿真并直接输出 torch 张量。
- 生成器：`agent/generator/` 支持规则生成与基于扩散模型的神经生成（`NetDemandGenerator`）。
- 规划器：`agent/planner/` 提供贪心、FRI、RBSO、DCP 等启发式，以及基于 DVRPNet 的 `ModelPlanner`。
- 控制器：`agent/controller/` 将目标队列转化为单步移动。
- 可视化：`utils/pygame_renderer.py` 支持 `--render` 实时查看。

---

## 环境安装

python -m venv .venv            # 可选：创建虚拟环境
source .venv/bin/activate       # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
# 需额外安装 PyTorch（根据你的 CUDA 版本）：https://pytorch.org/get-started/locally/
```

说明：`requirements.txt` 不包含 torch，请按本机 CUDA/CPU 情况单独安装；使用神经规划器/生成器与训练流程都需要 PyTorch。

---

### 对抗式训练（统一入口）
新增统一的对抗式训练脚本与 API（已合并到 adversarial 目录）：

- 入口：`adversarial/train_adversarial.py`
- 框架代码：`adversarial/` 包（`builders.py`, `trainers.py`）
- 文档：`docs/architecture.md`, `docs/api.md`

示例：

```bash
python adversarial/train_adversarial.py --episodes 50 --device cuda --planner greedy --randomize_depot
```

说明：
- 生成器使用扩散模型（可从 `checkpoints/diffusion_model.pth` 初始化）。
- 规划器支持 greedy 或 model（带 `--planner_ckpt`）。
- 环境碰撞策略统一为 first-wins：同一步多个代理竞争非仓库格时，索引最小者前进，其余回退。

相关旧路径已迁移：
`scripts/train_diffusion_generator.py` → `training/generator/train_diffusion_generator.py`；
`scripts/train_rl_diffusion_generator.py` → `training/generator/train_rl_diffusion_generator.py`；
`training/adversarial/train_adversarial.py` → `adversarial/train_adversarial.py`。
## 快速运行（`run.py`）
规则基线：

```bash
python run.py --seed 0
```

启用渲染（打开 Pygame 窗口）：

```bash
python run.py --seed 0 --render --fps 10
```

使用神经规划器与检查点：

```bash
python run.py --model --ckpt checkpoints/planner/planner_dynamic_20_2_200.pt --render
- `--planner`：`greedy|fri|rbso|dcp|model`；`--model` 等价于 `--planner model`；
- `--ckpt`：当使用 `model` 规划器时加载的权重路径（`.pt`）。
- `--cvrp` / `--cvrp-root` / `--cvrp-ckpt`：启用下述 CVRP-POMO 规划器集成。

注意：每个 episode 起始都会根据种子随机仓库位置，并同步更新生成器参数；各 agent 初始叠放在仓库，容量为满；非仓库位置自动避免碰撞。

### CVRP (POMO) 规划器适配

为了直接复用 CVRP POMO 项目的模型（`CVRPModel` + `CVRPEnv`），新增了 `CVRPPOMOPlanner` 以及对应配置 `Config.cvrp_planner_params`。该适配器会在每次重规划时将 DVRP 当前需求快照转换为 CVRP 静态实例，调用 POMO 模型求解，再按“回仓”分段为多辆车的目标队列。

使用步骤：

1. **准备 POMO 代码与权重**：将官方 `POMO/NEW_py_ver/CVRP/POMO` 目录放在本地，记录绝对路径；同时准备 `checkpoint-xxxxx.pt`（或 `saved_CVRP100_model` 内的 `checkpoint-30500.pt` 等）。
2. **配置路径**：在 `configs.py` 中设置 `cvrp_planner_params`（`pomo_root`、`checkpoint`、`env_params`、`model_params` 等），或在命令行通过 `--cvrp-root` / `--cvrp-ckpt` 临时覆盖。
3. **运行**：

```bash
python run.py --seed 0 --cvrp --cvrp-root ~/POMO-master/NEW_py_ver/CVRP/POMO \
  --cvrp-ckpt /path/to/saved_CVRP100_model/checkpoint-30500.pt
```

适配器要点：
- 仅支持 CPU 推理；CVRP 模型会保持在 `torch.device('cpu')`。
- 所有需求的 (x,y) 会按 `max(width, height)` 归一化到 `[0,1]`，容量按 `Config.capacity` 归一化为 1；请确保 `capacity` 不小于总需求。
- 若可视需求数超过 `cvrp_planner_params['max_nodes']`，会按 `selection_policy`（默认最早到期）选择子集。
- CVRP 求得的多段路径会依序分配给 DVRP 的车辆，路线末尾自动追加回仓节点。

---

## 项目结构（当前仓库）

```
configs.py              # 全局默认配置（地图尺寸、生成器/规划器参数等）
run.py                  # 主入口：构建环境+生成器+规划器+控制器并运行一局
run_all.sh              # 批量实验示例脚本
training/
  planner/
    data_gen.py                 # 为规划器监督训练生成 rows 数据集
    train_model.py              # DVRPNet 监督训练
    train_rl_planner.py         # DVRPNet 强化学习微调（REINFORCE/PPO）
  generator/
    generate_data.py            # 规则生成器的大规模离线数据（给扩散模型）
    normalize_data.py           # 将 CSV 归一化为 .pt（三份：train/val/test）
    train_diffusion_generator.py# 训练扩散式需求生成器
adversarial/
  train_adversarial.py             # 统一对抗式训练入口（生成器 vs 规划器）
agent/
  generator/            # 规则生成器、神经生成器与数据工具
  planner/              # 启发式/神经规划器实现
  controller/           # 单步控制策略
environment/
  env.py                # 网格环境与交互逻辑
models/
  generator_model/      # 扩散模型结构
  planner_model/        # DVRPNet 编码器/解码器
utils/
  pygame_renderer.py    # 可视化
  state_manager.py      # 规划状态管理（全局节点、计划队列等）
checkpoints/            # 现有权重（扩散/规划器）
runs/                   # 训练日志（TensorBoard、曲线等）
```

---

## 使用神经需求生成器（扩散模型）

1) 配置参数搜索范围：`configs.py` 的 `GENERATOR_PARAM_SPACE`。

2) 生成原始 CSV 数据（默认写入到 `training/generator/generate_data.py` 内置的 DATASET_DIR（示例设为 `/data/ruled_generator`），请按需修改为项目内相对路径 `data/ruled_generator` 或自定义目录）：
```bash
python training/generator/generate_data.py
```

3) 归一化并切分数据集（默认输出到 `data/ruled_generator/normalized_dataset_extended_diversity.pt`）：
```bash
python training/generator/normalize_data.py
```

4) 训练扩散生成器，保存到 `checkpoints/diffusion_model.pth`，日志写到 `runs/generator_training/`：
```bash
python training/generator/train_diffusion_generator.py
```

5) 使用神经生成器运行仿真：
- 方式 A：在 `configs.py` 将 `generator_type = "net"`（并确保 `generator_params.model_path` 指向上一步权重）；
- 方式 B：命令行覆盖：`python run.py --generator net`。

提示：较大的 batch（示例默认 6000）可提升稳定性但需更高显存；请结合硬件调小 `BATCH_SIZE`、增大积累步或降低维度。

### 扩散生成器对抗式 RL 训练（最小化规划器奖励）
希望探索“最难”需求分布，使某一规划器（贪心或 DVRPNet 模型规划器）在环境中获得最低奖励，可使用新脚本：
```bash
python training/adversarial/train_rl_diffusion_generator.py --episodes 50 --planner greedy --device cuda
```
或针对模型规划器：
```bash
python training/adversarial/train_rl_diffusion_generator.py --episodes 50 --planner model --planner_ckpt checkpoints/planner/planner_dynamic_20_2_200.pt --device cuda
```
从已有扩散监督权重初始化（提升稳定性）：
```bash
python training/adversarial/train_rl_diffusion_generator.py --episodes 50 \
  --planner greedy \
  --init_diffusion_ckpt checkpoints/diffusion_model.pth \
  --device cuda
```
启用实时渲染或保存帧：
```bash
python training/adversarial/train_rl_diffusion_generator.py --episodes 5 --planner greedy --render --fps 8
python training/adversarial/train_rl_diffusion_generator.py --episodes 5 --planner greedy --save_frames_dir adv_frames
```
机制要点：
- 生成器（扩散模型）一次生成整局需求集合，按照 `configs.py` 约束裁剪。
- 环境运行后取 episode 奖励 `R_env`，生成器优化目标为 `R_gen = -R_env`。
- 使用“奖励加权的噪声预测损失” (REINFORCE 风格)：`loss = diff_loss * advantage`，其中 baseline 采用 EMA 降低方差。
- 输出对抗权重：`checkpoints/diffusion_model_adv.pth`，日志：`runs/diffusion_adv_rewards.csv`。
 - 通过 `--init_diffusion_ckpt` 指定初始权重（例如原始监督训练的 `checkpoints/diffusion_model.pth`）。
 - 可选 `--render`/`--fps` 实时查看；或用 `--save_frames_dir` 在无显示环境保存 PNG 帧再用 ffmpeg 合成视频。
注意：此初版方法将整局视为单动作，不对单个需求的选择进行逐步 credit 分配；后续可扩展为逐点生成 + PPO。

---

## 训练 CVRP 规划器（两阶段）

### 阶段 A：静态快照（多车 CVRP）
1) 只在 `t=0` 放出所有需求，收集静态 rows（默认贪心导师，可改 `fri|rbso|dcp`）：
```bash
python training/planner/data_gen.py --episodes 200 --planner greedy --map_wid 20 --agent_num 2 \
  --out_dir data/static_rows --plan_horizon 8 --stage static
```
若想在静态阶段保留完整 episode（直到所有需求完成且车辆全部回仓），可额外加 `--static-full-episode`。
2) 训练静态模型（CVRP 编码器 + 多智能体解码器），得到 `planner_static_20_2_200.pt`：
```bash
python training/planner/train_model.py --map_wid 20 --agent_num 2 \
  --epochs 200 --device cuda --stage static --ckpt_dir checkpoints
```
默认会在 `data/static_rows` 中寻找 `plans_train/val_{map_wid}_{agent_num}.pt`，并将权重写入 `checkpoints/static_planner/`；你可以通过 `--data_dir`/`--prefix` 改为自定义路径，或者 `--model_config training/planner/model_config.json` 调整 `d_model/nhead/nlayers/adapter_dim` 等结构参数。
可选（启用 early stopping 并保存 val 最优模型）：
```bash
python training/planner/train_model.py --map_wid 20 --agent_num 2 \
  --prefix static_stage --epochs 200 --device cuda --stage static --ckpt_dir checkpoints \
  --early_stop --patience 20 --min_delta 1e-4 --save_best
```

### 阶段 B：动态适配（时间窗 + 实时释放）
1) 生成在线 rows（保持随机需求释放与时间窗）：
```bash
python training/planner/data_gen.py --episodes 200 --planner greedy --map_wid 20 --agent_num 2 \
  --out_dir data/dynamicrows --plan_horizon 8 --stage dynamic
```
2) 载入静态权重，为 DVRPNet 追加轻量 bottleneck adapter（默认 `adapter_dim=64`），并在需要时仅训练该适配器（`--freeze_base`），输出 `planner_dynamic_20_2_200.pt`：
```bash
python training/planner/train_model.py --map_wid 20 --agent_num 2 \
  --epochs 200 --device cuda --stage dynamic \
  --static_ckpt checkpoints/static_planner/planner_static_20_2_200.pt --ckpt_dir checkpoints
```
动态阶段会自动读取 `data/dynamicrows/plans_train_20_2.pt` / `...val...`，将结果写入 `checkpoints/dynamic_planner/`。当设置 `--freeze_base` 时，仅 residual adapter（包含时间窗偏置与 token/agent 两支瓶颈层）保持可训练，其余参数全部冻结；如需更宽的适配器，可通过 `--adapter_dim` 或 `training/planner/model_config.json` 调整。
可选（early stopping + 保存最优动态适配器）：
```bash
python training/planner/train_model.py --map_wid 20 --agent_num 2 \
  --prefix dynamic_stage --epochs 200 --device cuda --stage dynamic \
  --static_ckpt checkpoints/static_planner/planner_static_20_2_200.pt --freeze_base \
  --ckpt_dir checkpoints --early_stop --patience 15 --save_best
```

> `run_all.sh` 已自动编排上述 4 步：`bash run_all.sh` 即可依次生成静态/动态数据并训练两阶段模型。

### 强化学习微调（REINFORCE / PPO）
以动态阶段 checkpoint 热启动，在环境中继续在线优化：
```bash
python training/planner/train_rl_planner.py --episodes 200 --algo ppo \
  --ckpt_init checkpoints/planner/planner_dynamic_20_2_200.pt \
  --save_best checkpoints/planner/planner_rl_best.pt --device cuda
```
奖励曲线/CSV 将写入 `runs/` 目录。

如需让训练场景与测试保持同样的车辆数量，可加 `--num_agents 2`（或任意整数），会在构建环境前覆盖 `Config.num_agents` 与相关生成器设定。

调优 PPO critic 时，可追加：

```bash
python training/planner/train_rl_planner.py --algo ppo ... --critic_diag_dir run/planner
```

会在 `run/planner/critic_YYYYmmdd-HHMMSS/` 下生成 `critic_metrics.csv` 与 `critic_metrics.png`，包含 value loss、预测值/returns/advantage 的均值与方差，便于后续排查。

### 向量化 REINFORCE（TensorEnv + DVRPNet）
批量 rollout 使用 `TensorGridEnvironment` 在 GPU 上并行推进多局 episode，减少 Python 循环与环境重置开销：

```bash
python training/planner/train_batch_rl.py --batches 50 --batch_size 8 --device cuda --gen_val_data
```

要点：
- `--batch_size` 即并行环境数量；脚本会在单个 `step` 中同时计算 8 份行动与梯度贡献。
- 仍使用 DVRPNet + RuleBasedController 组合，并与克隆出的 baseline 模型比较得到 advantage。
- 依赖 `environment/env_tensor.py`，不会修改原有的 `GridEnvironment` 或旧版脚本。
- 验证阶段同样采用向量化，默认沿用训练时的 `--batch_size` 作为验证并行度，减少额外配置并保持显存占用一致。
奖励曲线/CSV 将写入 `runs/` 目录。

在 `run.py` 中使用训练好的规划器：
```bash
python run.py --planner model --ckpt checkpoints/planner/planner_dynamic_20_2_200.pt --render
```

### POMO 自对弈策略优化
为 DVRPNet 引入 Policy Optimization with Multiple Optima（POMO）策略：同一需求场景会运行多次独立采样，互为 baseline 以降低方差。

```bash
python training/planner/train_rl_planner.py \
  --episodes 100 \
  --algo pomo \
  --num_agents 2 \
  --ckpt_init checkpoints/planner/planner_dynamic_20_2_200.pt \
  --pomo_size 16 \
  --device cuda
```

关键开关：
- `--pomo_size`: 每个 episode 的自对弈副本数（默认 1，可设 8/16 等）；
- `--pomo_entropy_coef`: 额外熵正则，提升副本间探索；
- `--pomo_baseline`: `leave_one_out`（默认）或 `mean`，控制优势函数的基线；
- `--pomo_seed_stride`: 控制副本之间的 `torch` 随机种子间隔，避免采样相关。
- `--num_agents`: 与评估阶段对齐，避免训练/推理车辆数量不一致导致策略发散。

每个副本共享同一 demand 轨迹（环境随机数固定），仅策略采样不同；平均回报会写入 `runs/rl_rewards.csv`，最佳副本用于保存 checkpoint。

---

## 命令行与配置要点

- 运行入口：`run.py` 提供以下关键开关：
  - `--planner`: 选择 `greedy|fri|rbso|dcp|model`。
  - `--model`: `--planner model` 的别名。
  - `--generator`: 选择 `rule|net`；当为 `net` 时，从 `configs.py.generator_params.model_path` 读取扩散权重。
  - `--ckpt`: `model` 规划器的 `.pt` 路径；与 `--model` 联用。
  - `--render`/`--fps`: 可视化控制。
- 全局默认：见 `configs.py`；其中 `generator_params` 的 `"__MAX_TIME__"/"__depot__"` 占位符会在构造时自动替换为 `Config.max_time/depot`。

---

## 预训练与日志

- 已包含示例权重：
  - 扩散生成器：`checkpoints/diffusion_model.pth`
  - 规划器（监督）：`checkpoints/planner/planner_static_20_2_200.pt`
  - 规划器（动态适配）：`checkpoints/planner/planner_dynamic_20_2_200.pt`
  - 规划器（RL 最优）：`checkpoints/planner/planner_rl_best.pt`
- TensorBoard：`runs/generator_training/` 按时间戳生成子目录，可用 `tensorboard --logdir runs/generator_training` 查看。

---

## 常见问题（FAQ）

1) 没装 PyTorch？— 安装与 CUDA 相关，参考官网指令；CPU 也可运行但训练较慢。

2) 运行渲染报错或没有窗口？— Linux 需要可用的显示环境（本地桌面或 X11/Wayland 转发）。服务器环境可去掉 `--render`，或使用虚拟显示。

3) 生成器数据脚本默认写入 `/data/ruled_generator`？— 这是演示路径，请修改 `training/generator/generate_data.py` 中的 `DATASET_DIR`，推荐使用项目内 `data/ruled_generator`。

4) `run.py --planner model` 没加载权重？— 确认 `--ckpt` 指向存在的 `.pt` 文件；或在 `configs.py.model_planner_params.ckpt` 里预设。

5) 训练出现 NaN/Inf？— 降低学习率、检查标签容量可行性（数据生成时已做校验）、或打开 `train_model.py --debug` 打印诊断。

---

## 引用

若本项目对你的研究/产品有帮助，请在致谢中引用本仓库（DVRP-11.5）。

