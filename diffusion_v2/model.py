"""
VRPDiffusionPolicy: Nano-DiT 架构的 VRP 需求生成器

特性:
- 3层 Transformer Encoder (禁止 U-Net)
- Depot-Aware 相对位置编码: 每个节点包含到 depot 的相对位置信息
- 线性层变换节点特征 (非 patch embedding)
- AdaLN-Zero 注入时间步和全局条件 (更稳定的初始化)
- 解耦输出头: CoordHead (Sigmoid) + DemandHead (Softmax)

优化点:
- Depot-relative 编码: 学习需求点与 depot 的空间关系
- AdaLN-Zero: 使用零初始化的 gate 参数，训练更稳定
- 高效注意力: 优化内存使用，支持更大 batch
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ==============================================================================
# 模型超参数 (静态常量)
# ==============================================================================

# Nano-DiT 架构参数
HIDDEN_DIM: int = 256          # Transformer 隐藏维度
NUM_HEADS: int = 8             # 注意力头数 (从4增加到8，更细粒度的注意力)
NUM_LAYERS: int = 3            # Transformer 层数
MLP_RATIO: float = 4.0         # MLP 扩展比例
DROPOUT: float = 0.1           # Dropout 比例

# Diffusion 参数
NUM_DIFFUSION_STEPS: int = 1000  # 扩散步数
BETA_START: float = 1e-4         # beta 起始值
BETA_END: float = 0.02           # beta 结束值

# 输入输出维度
NODE_INPUT_DIM: int = 3          # 节点输入: (x, y, demand_logit)
COORD_OUTPUT_DIM: int = 2        # 坐标输出: (x, y)
DEMAND_OUTPUT_DIM: int = 1       # 需求输出: (demand_ratio)
GLOBAL_COND_DIM: int = 4         # 全局条件: (depot_x, depot_y, target_load_ratio, num_nodes_normalized)

# 时间嵌入维度
TIME_EMB_DIM: int = 128

# Depot-relative 编码维度
DEPOT_REL_DIM: int = 4           # 相对位置特征: (rel_x, rel_y, distance, angle)


# ==============================================================================
# 辅助模块
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步嵌入"""
    
    def __init__(self, dim: int = TIME_EMB_DIM):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (Batch,) 时间步
        Returns:
            emb: (Batch, dim) 时间嵌入
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DepotRelativeEncoder(nn.Module):
    """
    Depot-Relative 位置编码器
    
    为每个节点计算相对于 depot 的位置特征:
    - 相对坐标 (rel_x, rel_y)
    - 到 depot 的距离
    - 到 depot 的角度 (sin, cos 编码)
    
    这让模型能理解每个需求点与 depot 的空间关系
    """
    
    def __init__(self, output_dim: int = HIDDEN_DIM):
        super().__init__()
        # 输入: (rel_x, rel_y, distance, sin_angle, cos_angle) = 5 维
        self.proj = nn.Sequential(
            nn.Linear(5, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        
    def forward(
        self, 
        node_coords: torch.Tensor, 
        depot_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_coords: (Batch, N, 2) 节点坐标 [0, 1]
            depot_coords: (Batch, 2) depot 坐标 [0, 1]
            
        Returns:
            rel_encoding: (Batch, N, output_dim) 相对位置编码
        """
        # depot_coords: (B, 2) -> (B, 1, 2)
        depot = depot_coords.unsqueeze(1)
        
        # 相对坐标
        rel_xy = node_coords - depot  # (B, N, 2)
        
        # 到 depot 的距离
        distance = torch.norm(rel_xy, dim=-1, keepdim=True)  # (B, N, 1)
        
        # 角度编码 (使用 atan2 计算角度，然后 sin/cos 编码)
        angle = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])  # (B, N)
        sin_angle = torch.sin(angle).unsqueeze(-1)  # (B, N, 1)
        cos_angle = torch.cos(angle).unsqueeze(-1)  # (B, N, 1)
        
        # 组合特征
        rel_features = torch.cat([
            rel_xy,          # (B, N, 2)
            distance,        # (B, N, 1)
            sin_angle,       # (B, N, 1)
            cos_angle,       # (B, N, 1)
        ], dim=-1)  # (B, N, 5)
        
        return self.proj(rel_features)


class AdaLayerNormZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero-initialization (AdaLN-Zero)
    
    改进版 AdaLN：
    - 使用 gate 参数控制调制强度
    - 零初始化，训练初期行为接近标准 LayerNorm
    - 更稳定的梯度流动
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # 生成 scale (gamma), shift (beta), gate (alpha)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3),
        )
        # 零初始化：训练开始时 gate=0，相当于跳过这一层
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)
        
    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, N, hidden_dim) 输入特征
            cond: (Batch, cond_dim) 条件向量
        Returns:
            out: (Batch, N, hidden_dim) 调制后的特征
            gate: (Batch, 1, hidden_dim) 用于残差连接的 gate
        """
        # 生成调制参数
        params = self.cond_proj(cond)  # (B, hidden_dim * 3)
        scale, shift, gate = params.chunk(3, dim=-1)
        
        # 扩展维度
        scale = scale.unsqueeze(1)  # (B, 1, hidden_dim)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        
        # 应用调制
        out = self.norm(x) * (1 + scale) + shift
        
        return out, gate


class EfficientSelfAttention(nn.Module):
    """
    高效自注意力模块
    
    优化:
    - 合并 QKV 投影为单个线性层
    - 支持 FlashAttention (当可用时)
    - 优化内存布局
    """
    
    def __init__(
        self, 
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 合并 QKV 投影
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, N, hidden_dim)
        Returns:
            out: (Batch, N, hidden_dim)
        """
        B, N, C = x.shape
        
        # 合并计算 QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # 尝试使用 scaled_dot_product_attention (PyTorch 2.0+, 自动使用 FlashAttention)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # 手动计算注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            attn_out = attn @ v
        
        # 重塑输出
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(attn_out)
        
        return out


class TransformerBlock(nn.Module):
    """
    带 AdaLN-Zero 的 Transformer Block
    
    结构: AdaLN-Zero -> Self-Attention -> Gate*Residual -> AdaLN-Zero -> MLP -> Gate*Residual
    
    改进:
    - 使用 AdaLN-Zero 实现更稳定的训练
    - Gate 控制的残差连接
    - 高效自注意力
    """
    
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        mlp_ratio: float = MLP_RATIO,
        cond_dim: int = TIME_EMB_DIM + GLOBAL_COND_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # AdaLN-Zero for attention
        self.adaln_attn = AdaLayerNormZero(hidden_dim, cond_dim)
        
        # 高效自注意力
        self.attn = EfficientSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # AdaLN-Zero for MLP
        self.adaln_mlp = AdaLayerNormZero(hidden_dim, cond_dim)
        
        # MLP with GELU
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, N, hidden_dim) 节点特征
            cond: (Batch, cond_dim) 条件向量 (时间 + 全局条件)
        Returns:
            out: (Batch, N, hidden_dim) 输出特征
        """
        # Self-Attention with AdaLN-Zero
        x_norm, gate_attn = self.adaln_attn(x, cond)
        attn_out = self.attn(x_norm)
        x = x + gate_attn * attn_out  # Gated residual
        
        # MLP with AdaLN-Zero
        x_norm, gate_mlp = self.adaln_mlp(x, cond)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp * mlp_out  # Gated residual
        
        return x


class CoordHead(nn.Module):
    """
    坐标输出头
    
    输出经 Sigmoid 归一化到 [0, 1] 的坐标
    """
    
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, COORD_OUTPUT_DIM),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, N, hidden_dim)
        Returns:
            coords: (Batch, N, 2) in [0, 1]
        """
        return torch.sigmoid(self.proj(x))


class DemandHead(nn.Module):
    """
    需求输出头
    
    输出经 Softmax 归一化的需求比例分布
    """
    
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, DEMAND_OUTPUT_DIM),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, N, hidden_dim)
        Returns:
            demand_ratios: (Batch, N, 1), sum=1 per batch
        """
        logits = self.proj(x).squeeze(-1)  # (Batch, N)
        ratios = F.softmax(logits, dim=-1)  # (Batch, N)
        return ratios.unsqueeze(-1)  # (Batch, N, 1)


# ==============================================================================
# 主模型: VRPDiffusionPolicy
# ==============================================================================

class VRPDiffusionPolicy(nn.Module):
    """
    VRP 需求生成 Diffusion 模型 (Nano-DiT 架构)
    
    架构特点:
    - 3层 Transformer Encoder
    - Depot-Aware 相对位置编码: 理解需求点与 depot 的空间关系
    - 线性层节点变换
    - AdaLN-Zero 注入时间和全局条件 (更稳定)
    - 解耦输出头
    
    输入:
        noisy_state: (Batch, N, 3) 加噪节点状态
        timestep: (Batch,) 扩散时间步
        global_condition: (Batch, 4) 全局条件 [depot_x, depot_y, target_load_ratio, num_nodes_norm]
    
    输出:
        pred_coords: (Batch, N, 2) 预测坐标 [0, 1]
        pred_demand_ratios: (Batch, N, 1) 需求比例分布
    """
    
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        mlp_ratio: float = MLP_RATIO,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(TIME_EMB_DIM),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
        )
        
        # 全局条件嵌入 (depot + load_ratio + num_nodes)
        self.global_cond_embed = nn.Sequential(
            nn.Linear(GLOBAL_COND_DIM, TIME_EMB_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
        )
        
        # Depot-relative 位置编码器
        self.depot_rel_encoder = DepotRelativeEncoder(hidden_dim)
        
        # 节点输入投影 (线性层，非 patch embedding)
        self.node_proj = nn.Linear(NODE_INPUT_DIM, hidden_dim)
        
        # 融合层: 合并节点特征和 depot-relative 编码
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer Blocks
        cond_dim = TIME_EMB_DIM * 2  # time_emb + global_cond_emb
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 解耦输出头
        self.coord_head = CoordHead(hidden_dim)
        self.demand_head = DemandHead(hidden_dim)
        
        # 初始化
        self._init_weights()
        
        # 注册 Diffusion 参数
        self._register_diffusion_params()
        
    def _init_weights(self):
        """初始化权重"""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        
    def _register_diffusion_params(self):
        """注册 Diffusion 调度参数为 buffer"""
        betas = torch.linspace(BETA_START, BETA_END, NUM_DIFFUSION_STEPS)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
    def forward(
        self,
        noisy_state: torch.Tensor,
        timestep: torch.Tensor,
        global_condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            noisy_state: (Batch, N, 3) 加噪节点状态 [x, y, demand_logit]
            timestep: (Batch,) 扩散时间步 t ∈ [0, NUM_DIFFUSION_STEPS)
            global_condition: (Batch, 4) [depot_x, depot_y, target_load_ratio, num_nodes_norm]
                              或 (Batch, 3) [depot_x, depot_y, target_load_ratio] (兼容旧接口)
            
        Returns:
            pred_coords: (Batch, N, 2) 预测去噪坐标 [0, 1]
            pred_demand_ratios: (Batch, N, 1) 需求比例分布
        """
        batch_size, num_nodes, _ = noisy_state.shape
        
        # 兼容旧的 3 维条件
        if global_condition.shape[-1] == 3:
            # 添加 num_nodes_norm (归一化到 [0,1]，假设最大 100 节点)
            num_nodes_norm = torch.full(
                (batch_size, 1), 
                num_nodes / 100.0,
                device=global_condition.device,
                dtype=global_condition.dtype
            )
            global_condition = torch.cat([global_condition, num_nodes_norm], dim=-1)
        
        # 时间嵌入
        t_emb = self.time_embed(timestep)  # (Batch, TIME_EMB_DIM)
        
        # 全局条件嵌入
        g_emb = self.global_cond_embed(global_condition)  # (Batch, TIME_EMB_DIM)
        
        # 合并条件
        cond = torch.cat([t_emb, g_emb], dim=-1)  # (Batch, TIME_EMB_DIM * 2)
        
        # 提取 depot 坐标
        depot_coords = global_condition[:, :2]  # (Batch, 2)
        
        # 节点坐标 (从 noisy_state 提取)
        node_coords = noisy_state[..., :2]  # (Batch, N, 2)
        
        # Depot-relative 编码
        depot_rel = self.depot_rel_encoder(node_coords, depot_coords)  # (B, N, hidden)
        
        # 节点投影
        node_feat = self.node_proj(noisy_state)  # (Batch, N, hidden_dim)
        
        # 融合节点特征和 depot-relative 编码
        x = self.fusion(torch.cat([node_feat, depot_rel], dim=-1))  # (B, N, hidden)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x, cond)
            
        # Final Norm
        x = self.final_norm(x)
        
        # 解耦输出
        pred_coords = self.coord_head(x)  # (Batch, N, 2)
        pred_demand_ratios = self.demand_head(x)  # (Batch, N, 1)
        
        return pred_coords, pred_demand_ratios
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向扩散: 给数据加噪
        
        Args:
            x_start: (Batch, N, 3) 原始数据
            t: (Batch,) 时间步
            noise: (Batch, N, 3) 噪声，可选
            
        Returns:
            x_t: (Batch, N, 3) 加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_num_params(self) -> int:
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# 工具函数
# ==============================================================================

def create_global_condition(
    depot_x: float,
    depot_y: float,
    total_demand: float,
    capacity: float,
    batch_size: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    创建全局条件张量
    
    Args:
        depot_x: depot x 坐标 (归一化到 [0, 1])
        depot_y: depot y 坐标 (归一化到 [0, 1])
        total_demand: 总需求量
        capacity: 车辆容量
        batch_size: batch 大小
        device: 设备
        
    Returns:
        condition: (batch_size, 3) [depot_x, depot_y, target_load_ratio]
    """
    target_load_ratio = total_demand / capacity if capacity > 0 else 1.0
    cond = torch.tensor(
        [[depot_x, depot_y, target_load_ratio]],
        dtype=torch.float32,
        device=device,
    ).expand(batch_size, -1)
    return cond


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VRPDiffusionPolicy on {device}")
    
    model = VRPDiffusionPolicy().to(device)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # 测试输入
    batch_size = 4
    num_nodes = 20
    
    noisy_state = torch.randn(batch_size, num_nodes, 3, device=device)
    timestep = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=device)
    global_cond = create_global_condition(
        depot_x=0.5, depot_y=0.5,
        total_demand=60, capacity=30,
        batch_size=batch_size, device=device
    )
    
    # 前向传播
    pred_coords, pred_demand_ratios = model(noisy_state, timestep, global_cond)
    
    print(f"Input noisy_state: {noisy_state.shape}")
    print(f"Input timestep: {timestep.shape}")
    print(f"Input global_cond: {global_cond.shape}")
    print(f"Output pred_coords: {pred_coords.shape}, range: [{pred_coords.min():.3f}, {pred_coords.max():.3f}]")
    print(f"Output pred_demand_ratios: {pred_demand_ratios.shape}, sum per batch: {pred_demand_ratios.sum(dim=1).squeeze()}")
    
    # 测试 q_sample
    x_start = torch.rand(batch_size, num_nodes, 3, device=device)
    x_t = model.q_sample(x_start, timestep)
    print(f"q_sample output: {x_t.shape}")
    
    print("\n✓ VRPDiffusionPolicy test passed!")
