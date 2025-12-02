"""
Dynamic VRP Model - Static model + Residual Adapter.

For DVRP, we freeze the static model and train lightweight adapters that:
1. Inject time-dependent features
2. Adjust node representations based on urgency/deadlines
3. Can be trained with supervised learning or RL
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union
import math
import torch
import torch.nn as nn

from .static_model import StaticVRPModel, Encoder, Decoder, ResetState, StepState


class ResidualAdapter(nn.Module):
    """
    Lightweight bottleneck adapter for domain adaptation.
    
    Adds a small number of trainable parameters while keeping
    the static model frozen.
    """
    
    def __init__(
        self,
        d_model: int,
        adapter_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # learnable scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(self.up(self.act(self.down(x))))
        return x + self.scale * residual


class TimeEncoder(nn.Module):
    """Encode time-related features for dynamic demands."""
    
    def __init__(self, d_model: int, adapter_dim: int = 32):
        super().__init__()
        # Input: (time_now, time_remaining, urgency)
        self.encoder = nn.Sequential(
            nn.Linear(3, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, d_model),
        )

    def forward(
        self,
        time_now: torch.Tensor,
        deadline: torch.Tensor,
        time_norm: float = 100.0,
    ) -> torch.Tensor:
        """
        Args:
            time_now: (batch,) current time
            deadline: (batch, n_nodes) node deadlines
            time_norm: normalization factor
            
        Returns:
            time_features: (batch, n_nodes, d_model)
        """
        batch_size, n_nodes = deadline.shape
        
        # Normalize time
        t_now = (time_now / time_norm).unsqueeze(1).expand(-1, n_nodes)
        t_deadline = deadline / time_norm
        t_remaining = torch.clamp(t_deadline - t_now, min=0)
        
        # Urgency: inverse of time remaining (higher = more urgent)
        urgency = 1.0 / (t_remaining + 0.1)
        urgency = urgency / urgency.max(dim=1, keepdim=True)[0].clamp(min=1.0)
        
        # Stack features
        features = torch.stack([
            t_now,
            t_remaining,
            urgency,
        ], dim=-1)  # (batch, n_nodes, 3)
        
        return self.encoder(features)


class DynamicVRPModel(nn.Module):
    """
    Dynamic VRP Model = Static VRP Model + Adapters.
    
    The static model is frozen, and adapters are trained to handle:
    - Time-varying demands
    - Urgency/deadline constraints
    - Dynamic node arrivals
    """
    
    def __init__(
        self,
        static_model: StaticVRPModel,
        adapter_dim: int = 32,
        freeze_static: bool = True,
    ):
        super().__init__()
        self.static_model = static_model
        self.embedding_dim = static_model.embedding_dim
        self.adapter_dim = adapter_dim
        
        # Freeze static model if requested
        if freeze_static:
            for param in self.static_model.parameters():
                param.requires_grad = False
        
        # Adapters for dynamic features
        self.node_adapter = ResidualAdapter(self.embedding_dim, adapter_dim)
        self.context_adapter = ResidualAdapter(self.embedding_dim, adapter_dim)
        self.time_encoder = TimeEncoder(self.embedding_dim, adapter_dim)
        
        # Additional decoder head for multi-agent assignment
        self.agent_embed = nn.Sequential(
            nn.Linear(4, adapter_dim),  # (x, y, load, time)
            nn.GELU(),
            nn.Linear(adapter_dim, self.embedding_dim),
        )
        
        # Cache
        self.encoded_nodes: Optional[torch.Tensor] = None
        self.adapted_nodes: Optional[torch.Tensor] = None

    def encode(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        node_deadline: Optional[torch.Tensor] = None,
        time_now: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode problem with dynamic adaptations.
        
        Args:
            depot_xy: (batch, 1, 2)
            node_xy: (batch, n_nodes, 2)
            node_demand: (batch, n_nodes)
            node_deadline: (batch, n_nodes) optional deadlines
            time_now: (batch,) current time
            
        Returns:
            encoded: (batch, n_nodes+1, embedding_dim)
        """
        # Static encoding
        node_xy_demand = torch.cat((node_xy, node_demand.unsqueeze(-1)), dim=-1)
        self.encoded_nodes = self.static_model.encoder(depot_xy, node_xy_demand)
        
        # Apply node adapter
        self.adapted_nodes = self.node_adapter(self.encoded_nodes)
        
        # Add time features if available
        if node_deadline is not None and time_now is not None:
            # Pad deadline for depot (set to large value)
            batch_size = depot_xy.size(0)
            depot_deadline = torch.ones(batch_size, 1, device=depot_xy.device) * 1000
            full_deadline = torch.cat([depot_deadline, node_deadline], dim=1)
            
            time_features = self.time_encoder(time_now, full_deadline)
            self.adapted_nodes = self.adapted_nodes + time_features
        
        # Set decoder KV
        self.static_model.decoder.set_kv(self.adapted_nodes)
        
        return self.adapted_nodes

    def decode_step(
        self,
        agent_states: torch.Tensor,
        ninf_mask: torch.Tensor,
        current_nodes: Optional[torch.Tensor] = None,
        return_full_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode one step for multi-agent assignment.
        
        Args:
            agent_states: (batch, n_agents, 4) - (x, y, load, time)
            ninf_mask: (batch, n_agents, n_nodes+1)
            current_nodes: (batch, n_agents) current node indices
            return_full_probs: If True, return full probability distribution
            
        Returns:
            selected: (batch, n_agents) selected node indices
            probs: (batch, n_agents) selected probabilities OR 
                   (batch, n_agents, n_nodes+1) full probs if return_full_probs=True
        """
        batch_size, n_agents, _ = agent_states.shape
        
        # Embed agent states
        agent_emb = self.agent_embed(agent_states)
        agent_emb = self.context_adapter(agent_emb)
        
        # Get node embeddings for current positions
        if current_nodes is not None:
            gathering_index = current_nodes[:, :, None].expand(-1, -1, self.embedding_dim)
            current_emb = self.adapted_nodes.unsqueeze(1).expand(-1, n_agents, -1, -1)
            current_emb = current_emb.gather(2, gathering_index.unsqueeze(2)).squeeze(2)
            agent_emb = agent_emb + current_emb
        
        # Compute load ratios
        load_ratio = agent_states[:, :, 2]  # assuming load is at index 2
        
        # Decode - use custom method to get full probs if needed
        if return_full_probs:
            selected, full_probs = self._decode_with_full_probs(agent_emb, load_ratio, ninf_mask)
            return selected, full_probs
        else:
            selected, prob = self.static_model.decoder(agent_emb, load_ratio, ninf_mask)
            return selected, prob

    def _decode_with_full_probs(
        self,
        encoded_last_node: torch.Tensor,
        load: torch.Tensor,
        ninf_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode step returning full probability distribution."""
        import math
        decoder = self.static_model.decoder
        
        # Concatenate last node embedding with load
        input_cat = torch.cat((encoded_last_node, load.unsqueeze(-1)), dim=-1)
        
        # Attention
        q_last = decoder._reshape_by_heads(decoder.Wq_last(input_cat))
        attn = decoder._multi_head_attention(q_last, decoder.k, decoder.v, ninf_mask)
        mh_out = decoder.multi_head_combine(attn)
        
        # Score calculation
        score = torch.matmul(mh_out, decoder.single_head_key) / decoder.sqrt_embedding_dim
        score = decoder.logit_clipping * torch.tanh(score)
        score = score + ninf_mask
        
        probs = torch.softmax(score, dim=-1)
        
        # Sample or argmax based on training mode
        if self.training:
            selected = probs.reshape(-1, probs.size(-1)).multinomial(1).reshape(
                probs.size(0), probs.size(1)
            )
        else:
            selected = probs.argmax(dim=-1)
        
        return selected, probs

    def forward_with_full_probs(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        agent_states: torch.Tensor,
        ninf_mask: torch.Tensor,
        node_deadline: Optional[torch.Tensor] = None,
        time_now: Optional[torch.Tensor] = None,
        current_nodes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass returning complete probability distributions.
        
        Returns:
            selected: (batch, n_agents)
            probs: (batch, n_agents, n_nodes+1) full probability distribution
        """
        # Encode
        self.encode(depot_xy, node_xy, node_demand, node_deadline, time_now)
        
        # Decode with full probs
        return self.decode_step(agent_states, ninf_mask, current_nodes, return_full_probs=True)

    def forward(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        agent_states: torch.Tensor,
        ninf_mask: torch.Tensor,
        node_deadline: Optional[torch.Tensor] = None,
        time_now: Optional[torch.Tensor] = None,
        current_nodes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            depot_xy: (batch, 1, 2)
            node_xy: (batch, n_nodes, 2)
            node_demand: (batch, n_nodes)
            agent_states: (batch, n_agents, 4)
            ninf_mask: (batch, n_agents, n_nodes+1)
            node_deadline: (batch, n_nodes) optional
            time_now: (batch,) optional
            current_nodes: (batch, n_agents) optional
            
        Returns:
            selected: (batch, n_agents)
            probs: (batch, n_agents, n_nodes+1)
        """
        # Encode
        self.encode(depot_xy, node_xy, node_demand, node_deadline, time_now)
        
        # Decode
        return self.decode_step(agent_states, ninf_mask, current_nodes)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only adapter parameters for training."""
        params = []
        params.extend(self.node_adapter.parameters())
        params.extend(self.context_adapter.parameters())
        params.extend(self.time_encoder.parameters())
        params.extend(self.agent_embed.parameters())
        return params

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict for adapters only."""
        return {
            'node_adapter': self.node_adapter.state_dict(),
            'context_adapter': self.context_adapter.state_dict(),
            'time_encoder': self.time_encoder.state_dict(),
            'agent_embed': self.agent_embed.state_dict(),
        }

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load adapter weights."""
        self.node_adapter.load_state_dict(state_dict['node_adapter'])
        self.context_adapter.load_state_dict(state_dict['context_adapter'])
        self.time_encoder.load_state_dict(state_dict['time_encoder'])
        self.agent_embed.load_state_dict(state_dict['agent_embed'])


def create_dynamic_model(
    static_model_or_checkpoint: Optional[Union[str, 'StaticVRPModel']] = None,
    embedding_dim: int = 128,
    encoder_layers: int = 6,
    heads: int = 8,
    qkv_dim: int = 16,
    ff_hidden: int = 512,
    adapter_dim: int = 32,
    freeze_static: bool = True,
    device: str = "cpu",
) -> DynamicVRPModel:
    """
    Factory function to create dynamic VRP model.
    
    Args:
        static_model_or_checkpoint: pretrained static model or path to checkpoint
        embedding_dim: model dimension
        encoder_layers: number of encoder layers
        heads: number of attention heads
        qkv_dim: dimension per head
        ff_hidden: feedforward hidden dimension
        adapter_dim: adapter bottleneck dimension
        freeze_static: whether to freeze static model
        device: device to load model on
        
    Returns:
        DynamicVRPModel instance
    """
    from .static_model import create_static_model, StaticVRPModel
    
    # Handle static model - either use provided model or create new one
    if static_model_or_checkpoint is None:
        # Create new static model from scratch
        static_model = create_static_model(
            embedding_dim=embedding_dim,
            encoder_layers=encoder_layers,
            heads=heads,
            qkv_dim=qkv_dim,
            ff_hidden=ff_hidden,
        )
    elif isinstance(static_model_or_checkpoint, StaticVRPModel):
        # Use provided model directly
        static_model = static_model_or_checkpoint
    elif isinstance(static_model_or_checkpoint, str):
        # Load from checkpoint
        static_model = create_static_model(
            embedding_dim=embedding_dim,
            encoder_layers=encoder_layers,
            heads=heads,
            qkv_dim=qkv_dim,
            ff_hidden=ff_hidden,
        )
        checkpoint = torch.load(static_model_or_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            static_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            static_model.load_state_dict(checkpoint['model'])
        else:
            static_model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported type for static_model_or_checkpoint: {type(static_model_or_checkpoint)}")
    
    # Create dynamic model
    dynamic_model = DynamicVRPModel(
        static_model=static_model,
        adapter_dim=adapter_dim,
        freeze_static=freeze_static,
    )
    
    return dynamic_model.to(device)