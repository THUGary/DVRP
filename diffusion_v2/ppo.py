"""
PPOAgent: PPO 强化学习代理

将 Diffusion 模型视为 Actor，使用 PPO Clip Loss 更新
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .model import VRPDiffusionPolicy

# ==============================================================================
# PPO 超参数 (静态常量)
# ==============================================================================

# PPO 参数
PPO_CLIP_EPSILON: float = 0.2         # PPO clip 范围
PPO_VALUE_COEF: float = 0.5           # value loss 系数
PPO_ENTROPY_COEF: float = 0.01        # 熵正则化系数
PPO_MAX_GRAD_NORM: float = 0.5        # 梯度裁剪
PPO_GAE_LAMBDA: float = 0.95          # GAE lambda
PPO_GAMMA: float = 0.99               # 折扣因子

# 训练参数
DEFAULT_LR: float = 3e-4              # 默认学习率
DEFAULT_BATCH_SIZE: int = 32          # 默认 batch 大小
DEFAULT_UPDATE_EPOCHS: int = 4        # 每次更新的 epoch 数

# 奖励归一化
REWARD_SCALE: float = 1.0             # 奖励缩放
REWARD_CLIP: float = 10.0             # 奖励裁剪范围


@dataclass
class RolloutBuffer:
    """
    Rollout 数据缓冲区
    
    存储用于 PPO 更新的数据
    """
    # 输入数据
    noisy_states: List[torch.Tensor] = field(default_factory=list)
    timesteps: List[torch.Tensor] = field(default_factory=list)
    global_conditions: List[torch.Tensor] = field(default_factory=list)
    
    # 输出数据
    pred_coords: List[torch.Tensor] = field(default_factory=list)
    pred_demand_ratios: List[torch.Tensor] = field(default_factory=list)
    
    # RL 数据
    rewards: List[float] = field(default_factory=list)
    old_log_probs: List[torch.Tensor] = field(default_factory=list)
    
    def add(
        self,
        noisy_state: torch.Tensor,
        timestep: torch.Tensor,
        global_condition: torch.Tensor,
        pred_coord: torch.Tensor,
        pred_demand_ratio: torch.Tensor,
        reward: float,
        old_log_prob: torch.Tensor,
    ):
        """添加一条数据"""
        self.noisy_states.append(noisy_state.detach())
        self.timesteps.append(timestep.detach())
        self.global_conditions.append(global_condition.detach())
        self.pred_coords.append(pred_coord.detach())
        self.pred_demand_ratios.append(pred_demand_ratio.detach())
        self.rewards.append(reward)
        self.old_log_probs.append(old_log_prob.detach())
    
    def clear(self):
        """清空缓冲区"""
        self.noisy_states.clear()
        self.timesteps.clear()
        self.global_conditions.clear()
        self.pred_coords.clear()
        self.pred_demand_ratios.clear()
        self.rewards.clear()
        self.old_log_probs.clear()
    
    def __len__(self):
        return len(self.rewards)
    
    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """获取批量数据"""
        if len(self) == 0:
            return {}
        
        # Stack all tensors
        batch = {
            "noisy_states": torch.stack(self.noisy_states).to(device),
            "timesteps": torch.stack(self.timesteps).to(device),
            "global_conditions": torch.stack(self.global_conditions).to(device),
            "pred_coords": torch.stack(self.pred_coords).to(device),
            "pred_demand_ratios": torch.stack(self.pred_demand_ratios).to(device),
            "rewards": torch.tensor(self.rewards, device=device, dtype=torch.float32),
            "old_log_probs": torch.stack(self.old_log_probs).to(device),
        }
        
        return batch


class PPOAgent:
    """
    PPO 强化学习代理
    
    将 Diffusion 模型视为 Actor，使用 PPO Clip Loss 更新
    
    特点:
    - 简化版 PPO (无 Critic 网络，使用奖励归一化替代)
    - 支持 Diffusion 模型的特殊结构
    - 使用生成的坐标/需求的高斯对数概率
    """
    
    def __init__(
        self,
        model: "VRPDiffusionPolicy",
        lr: float = DEFAULT_LR,
        clip_epsilon: float = PPO_CLIP_EPSILON,
        entropy_coef: float = PPO_ENTROPY_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        update_epochs: int = DEFAULT_UPDATE_EPOCHS,
    ):
        """
        Args:
            model: VRPDiffusionPolicy 模型
            lr: 学习率
            clip_epsilon: PPO clip 范围
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪
            update_epochs: 每次更新的 epoch 数
        """
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        
        # 奖励归一化统计
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        self.device = next(model.parameters()).device
    
    def compute_log_prob(
        self,
        pred_coords: torch.Tensor,
        pred_demand_ratios: torch.Tensor,
        target_coords: torch.Tensor,
        target_demand_ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算动作的对数概率
        
        假设坐标服从以预测为均值的高斯分布
        需求比例使用 categorical 分布
        
        Args:
            pred_coords: (B, N, 2) 预测坐标
            pred_demand_ratios: (B, N, 1) 预测需求比例
            target_coords: (B, N, 2) 目标坐标
            target_demand_ratios: (B, N, 1) 目标需求比例
            
        Returns:
            log_prob: (B,) 对数概率
        """
        # 坐标的高斯对数概率 (固定 std=0.1)
        coord_std = 0.1
        coord_diff = (target_coords - pred_coords) / coord_std
        coord_log_prob = -0.5 * (coord_diff ** 2).sum(dim=(-1, -2))  # (B,)
        
        # 需求比例的交叉熵 (softmax 输出视为概率)
        # 使用 KL 散度的负值作为对数概率近似
        demand_log_prob = -(
            target_demand_ratios * 
            (target_demand_ratios.log() - pred_demand_ratios.log() + 1e-8)
        ).sum(dim=(-1, -2))  # (B,)
        
        return coord_log_prob + demand_log_prob
    
    def collect_rollout(
        self,
        noisy_state: torch.Tensor,
        timestep: torch.Tensor,
        global_condition: torch.Tensor,
        reward: float,
    ):
        """
        收集一条 rollout 数据
        
        Args:
            noisy_state: (1, N, 3) 噪声状态
            timestep: (1,) 时间步
            global_condition: (1, 3) 全局条件
            reward: 奖励值
        """
        self.model.eval()
        with torch.no_grad():
            pred_coords, pred_demand_ratios = self.model(
                noisy_state, timestep, global_condition
            )
            
            # 计算旧的对数概率
            old_log_prob = self.compute_log_prob(
                pred_coords, pred_demand_ratios,
                pred_coords, pred_demand_ratios,  # 使用预测作为目标
            )
        
        self.buffer.add(
            noisy_state=noisy_state.squeeze(0),
            timestep=timestep.squeeze(0),
            global_condition=global_condition.squeeze(0),
            pred_coord=pred_coords.squeeze(0),
            pred_demand_ratio=pred_demand_ratios.squeeze(0),
            reward=reward,
            old_log_prob=old_log_prob.squeeze(0),
        )
    
    def update_reward_stats(self, rewards: List[float]):
        """更新奖励归一化统计"""
        for r in rewards:
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_std = (
                (self.reward_count - 1) * self.reward_std ** 2 + delta * delta2
            ) / self.reward_count
            self.reward_std = max(self.reward_std ** 0.5, 1e-8)
    
    def normalize_reward(self, reward: float) -> float:
        """归一化奖励"""
        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return max(-REWARD_CLIP, min(REWARD_CLIP, normalized))
    
    def update(self) -> Dict[str, float]:
        """
        执行 PPO 更新
        
        Returns:
            metrics: 包含损失等指标的字典
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "entropy": 0.0}
        
        self.model.train()
        
        # 获取批量数据
        batch = self.buffer.get_batch(self.device)
        
        # 更新奖励统计并归一化
        rewards = batch["rewards"]
        self.update_reward_stats(rewards.tolist())
        normalized_rewards = torch.tensor(
            [self.normalize_reward(r.item()) for r in rewards],
            device=self.device,
            dtype=torch.float32,
        )
        
        # 计算优势 (简化版: 直接使用归一化奖励)
        advantages = normalized_rewards
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        # 多次更新
        for _ in range(self.update_epochs):
            # 前向传播
            pred_coords, pred_demand_ratios = self.model(
                batch["noisy_states"],
                batch["timesteps"],
                batch["global_conditions"],
            )
            
            # 计算新的对数概率
            new_log_probs = self.compute_log_prob(
                pred_coords, pred_demand_ratios,
                batch["pred_coords"], batch["pred_demand_ratios"],
            )
            
            # 计算比率
            old_log_probs = batch["old_log_probs"]
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO Clip Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 熵正则化 (鼓励探索)
            # 使用预测的不确定性作为熵的近似
            coord_entropy = 0.5 * torch.log(torch.tensor(2 * 3.14159 * 2.71828 * 0.1 ** 2, device=self.device)) * pred_coords.numel() / len(pred_coords)
            demand_entropy = -(pred_demand_ratios * pred_demand_ratios.log().clamp(min=-10)).sum(dim=(-1, -2)).mean()
            entropy = coord_entropy + demand_entropy
            
            # 总损失
            loss = policy_loss - self.entropy_coef * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
            num_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
        }
    
    def save(self, path: str):
        """保存检查点"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_count": self.reward_count,
        }, path)
    
    def load(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.reward_mean = checkpoint.get("reward_mean", 0.0)
        self.reward_std = checkpoint.get("reward_std", 1.0)
        self.reward_count = checkpoint.get("reward_count", 0)


if __name__ == "__main__":
    # 测试 PPO Agent
    from .model import VRPDiffusionPolicy, create_global_condition, NUM_DIFFUSION_STEPS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PPOAgent on {device}")
    
    # 创建模型和代理
    model = VRPDiffusionPolicy().to(device)
    agent = PPOAgent(model, lr=3e-4)
    
    # 测试参数
    batch_size = 8
    num_nodes = 20
    
    # 模拟收集 rollout
    for i in range(batch_size):
        noisy_state = torch.randn(1, num_nodes, 3, device=device)
        timestep = torch.randint(0, NUM_DIFFUSION_STEPS, (1,), device=device)
        global_cond = create_global_condition(
            depot_x=0.5, depot_y=0.5,
            total_demand=60, capacity=30,
            batch_size=1, device=device
        )
        
        # 模拟奖励
        reward = torch.randn(1).item() * 0.5 + 0.5
        
        agent.collect_rollout(noisy_state, timestep, global_cond, reward)
    
    print(f"Buffer size: {len(agent.buffer)}")
    
    # 执行更新
    metrics = agent.update()
    
    print(f"\nUpdate metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试保存/加载
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pth")
        agent.save(ckpt_path)
        print(f"\nSaved checkpoint to {ckpt_path}")
        
        # 创建新代理并加载
        new_model = VRPDiffusionPolicy().to(device)
        new_agent = PPOAgent(new_model)
        new_agent.load(ckpt_path)
        print(f"Loaded checkpoint, reward_mean={new_agent.reward_mean:.4f}")
    
    print("\n✓ PPOAgent test passed!")
