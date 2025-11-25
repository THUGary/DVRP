from __future__ import annotations
from typing import Dict, Optional

import math
import torch
import torch.nn as nn


def prepare_features(
    *,
    nodes,
    node_mask,
    depot,
    d_model: int = 128,
    device: str | torch.device = "cpu",
    time_now: Optional[torch.Tensor | float | int | list | tuple] = None,
) -> Dict[str, torch.Tensor]:
    """Convert raw snapshot tensors into the format consumed by DVRPNet."""
    dev = torch.device(device)

    if isinstance(nodes, torch.Tensor):
        nodes_t = nodes.to(dev)
        if nodes_t.dim() == 2:
            nodes_t = nodes_t.unsqueeze(0)
    else:
        nodes_t = torch.tensor(nodes, dtype=torch.float32, device=dev)
        if nodes_t.dim() == 2:
            nodes_t = nodes_t.unsqueeze(0)

    if isinstance(node_mask, torch.Tensor):
        mask_t = node_mask.to(dev)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
    else:
        mask_t = torch.tensor(node_mask, dtype=torch.bool, device=dev)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
    mask_t = mask_t.bool()

    if isinstance(depot, torch.Tensor):
        depot_t = depot.to(dev)
        if depot_t.dim() == 2:
            depot_t = depot_t.unsqueeze(0)
    else:
        depot_t = torch.tensor(depot, dtype=torch.float32, device=dev)
        if depot_t.dim() == 2:
            depot_t = depot_t.unsqueeze(0)
    depot_t = depot_t[..., :2]

    B = nodes_t.size(0)

    if time_now is None:
        time_t = torch.zeros(B, dtype=torch.float32, device=dev)
    elif isinstance(time_now, torch.Tensor):
        time_t = time_now.to(dev).float()
        if time_t.dim() == 0:
            time_t = time_t.expand(B)
        elif time_t.dim() == 1 and time_t.size(0) == 1 and B > 1:
            time_t = time_t.expand(B)
    else:
        time_t = torch.tensor(time_now, dtype=torch.float32, device=dev)
        if time_t.dim() == 0:
            time_t = time_t.expand(B)
        elif time_t.dim() == 1 and time_t.size(0) == 1 and B > 1:
            time_t = time_t.expand(B)
    if time_t.dim() == 1 and time_t.size(0) != B:
        raise ValueError(f"time_now batch ({time_t.size(0)}) does not match nodes batch ({B})")

    return {"nodes": nodes_t, "node_mask": mask_t, "depot": depot_t, "time_now": time_t}


def prepare_agents(agents, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert agent states to [B, A, 4] float tensor."""
    dev = torch.device(device)
    if isinstance(agents, torch.Tensor):
        t = agents.to(dev)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t
    t = torch.tensor(agents, dtype=torch.float32, device=dev)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


class CVRPStyleEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        encoder_layer_num: int = 6,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim) for _ in range(encoder_layer_num)]
        )

    def forward(self, depot_xy: torch.Tensor, node_xy_demand: torch.Tensor) -> torch.Tensor:
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int, ff_hidden_dim: int) -> None:
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_norm_1 = AddAndInstanceNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_norm_2 = AddAndInstanceNormalization(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        out_concat = multi_head_attention(q, k, v)
        mh_out = self.multi_head_combine(out_concat)
        out1 = self.add_norm_1(x, mh_out)
        ff = self.feed_forward(out1)
        out2 = self.add_norm_2(out1, ff)
        return out2


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        # LayerNorm avoids the InstanceNorm constraint that needs >1 spatial element
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        added = input1 + input2
        return self.norm(added)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_hidden_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(torch.relu(self.W1(x)))


class ResidualAdapter(nn.Module):
    """Lightweight bottleneck adapter trained in the dynamic stage only."""

    def __init__(self, d_model: int, adapter_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(adapter_dim, d_model)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(self.up(self.act(self.down(x))))
        return x + residual


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)


def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank3_ninf_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / math.sqrt(q.size(-1))
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :]
    weights = torch.softmax(score_scaled, dim=3)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(q.size(0), q.size(2), -1)


class MultiAgentDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        logit_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        self.head_num = head_num
        self.sqrt_embedding_dim = math.sqrt(embedding_dim)
        self.logit_clipping = logit_clipping
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.single_head_key: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def set_kv(self, encoded_tokens: torch.Tensor) -> None:
        self.k = reshape_by_heads(self.Wk(encoded_tokens), self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_tokens), self.head_num)
        self.single_head_key = encoded_tokens.transpose(1, 2)

    def forward(
        self,
        agent_emb: torch.Tensor,
        load_ratio: torch.Tensor,
        ninf_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.k is None or self.v is None or self.single_head_key is None:
            raise RuntimeError("Decoder KV cache not initialized. Call set_kv() before forward().")
        input_cat = torch.cat((agent_emb, load_ratio.unsqueeze(-1)), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), self.head_num)
        attn = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_out = self.multi_head_combine(attn)
        score = torch.matmul(mh_out, self.single_head_key)
        score = score / self.sqrt_embedding_dim
        score = self.logit_clipping * torch.tanh(score)
        return score + ninf_mask


class DVRPNet(nn.Module):
    """CVRP-style encoder/decoder with optional dynamic adapter."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        nlayers: int = 6,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
        coord_norm: float = 20.0,
        capacity_norm: float = 200.0,
        time_norm: float = 100.0,
        adapter_dim: int = 0,
        logit_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.coord_norm = max(coord_norm, 1.0)
        self.capacity_norm = max(capacity_norm, 1.0)
        self.time_norm = max(time_norm, 1.0)
        self.adapter_dim = max(0, int(adapter_dim))
        self.encoder = CVRPStyleEncoder(
            embedding_dim=d_model,
            encoder_layer_num=nlayers,
            head_num=nhead,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.decoder = MultiAgentDecoder(
            embedding_dim=d_model,
            head_num=nhead,
            qkv_dim=qkv_dim,
            logit_clipping=logit_clipping,
        )
        self.agent_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        if self.adapter_dim > 0:
            self.time_adapter = nn.Sequential(
                nn.Linear(2, self.adapter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.adapter_dim, d_model),
            )
            self.feature_adapter = ResidualAdapter(d_model, self.adapter_dim)
            self.agent_adapter = ResidualAdapter(d_model, self.adapter_dim)
        else:
            self.time_adapter = None
            self.feature_adapter = None
            self.agent_adapter = None

    @torch.no_grad()
    def encode(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nodes = feats["nodes"]
        node_mask = feats["node_mask"].bool()
        depot = feats["depot"]
        time_now = feats.get("time_now")
        node_xy = nodes[..., :2] / self.coord_norm
        demand = nodes[..., 3:4] / self.capacity_norm
        depot_xy = depot / self.coord_norm
        tokens = self.encoder(depot_xy, torch.cat([node_xy, demand], dim=-1))
        H_depot = tokens[:, :1, :]
        H_nodes = tokens[:, 1:, :]
        if self.time_adapter is not None and time_now is not None:
            H_nodes = self._apply_time_adapter(H_nodes, nodes, time_now)
        if self.feature_adapter is not None:
            H_nodes = self.feature_adapter(H_nodes)
            H_depot = self.feature_adapter(H_depot)
        return {"H_nodes": H_nodes, "H_depot": H_depot, "node_mask": node_mask}

    def _apply_time_adapter(self, H_nodes: torch.Tensor, nodes: torch.Tensor, time_now: torch.Tensor) -> torch.Tensor:
        B, N, _ = nodes.shape
        if N == 0:
            return H_nodes
        time_vec = time_now.view(B, 1, 1).expand(B, N, 1) / self.time_norm
        arrival = nodes[..., 2:3] / self.time_norm
        due = nodes[..., 4:5] / self.time_norm
        delta = torch.cat([time_vec - arrival, due - time_vec], dim=-1)
        return H_nodes + self.time_adapter(delta)

    def _encode_agents(self, agents_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ax = agents_tensor[..., 0:1] / self.coord_norm
        ay = agents_tensor[..., 1:2] / self.coord_norm
        load_ratio = torch.clamp(agents_tensor[..., 2:3] / self.capacity_norm, 0.0, 1.0)
        time_ratio = agents_tensor[..., 3:4] / self.time_norm
        feats = torch.cat([ax, ay, load_ratio, time_ratio], dim=-1)
        emb = self.agent_embed(feats)
        if self.agent_adapter is not None:
            emb = self.agent_adapter(emb)
        return emb, load_ratio.squeeze(-1)

    def _build_ninf_mask(
        self,
        node_mask: torch.Tensor,
        agents_tensor: torch.Tensor,
        nodes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, A = agents_tensor.shape[:2]
        N = node_mask.size(1)
        device = agents_tensor.device
        dtype = agents_tensor.dtype
        neg_inf = torch.finfo(dtype).min
        mask = torch.zeros(B, A, N + 1, device=device, dtype=dtype)
        if N > 0:
            base = node_mask[:, None, :].expand(B, A, N)
            mask[..., 1:] = torch.where(base, neg_inf, 0.0)
            if nodes is not None:
                demand = nodes[..., 3].unsqueeze(1).expand(B, A, N)
                cap = agents_tensor[..., 2].unsqueeze(-1)
                infeasible = demand > cap + 1e-6
                mask[..., 1:] = torch.where(infeasible, neg_inf, mask[..., 1:])
        return mask

    def decode(
        self,
        *,
        enc_nodes: torch.Tensor,
        enc_depot: torch.Tensor,
        node_mask: torch.Tensor,
        agents_tensor: torch.Tensor,
        nodes: Optional[torch.Tensor] = None,
        lateness_lambda: float = 0.0,
        history_positions: Optional[torch.Tensor] = None,
        history_target_coords: Optional[torch.Tensor] = None,
        time_now: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if agents_tensor is None:
            raise ValueError("decode requires agents_tensor with shape [B,A,4]")
        kv = torch.cat([enc_depot, enc_nodes], dim=1)
        self.decoder.set_kv(kv)
        agent_emb, load_ratio = self._encode_agents(agents_tensor)
        ninf_mask = self._build_ninf_mask(node_mask, agents_tensor, nodes)
        logits = self.decoder(agent_emb, load_ratio, ninf_mask)
        if nodes is not None and lateness_lambda > 0:
            B, N, _ = nodes.shape
            if N > 0:
                ax = agents_tensor[..., 0].unsqueeze(-1)
                ay = agents_tensor[..., 1].unsqueeze(-1)
                nx = nodes[..., 0].unsqueeze(1)
                ny = nodes[..., 1].unsqueeze(1)
                dist = (ax - nx).abs() + (ay - ny).abs()
                t_agent = agents_tensor[..., 3].unsqueeze(-1)
                t_due = nodes[..., 4].unsqueeze(1)
                lateness = torch.clamp(t_agent + dist - t_due, min=0.0)
                logits[..., 1 : N + 1] = logits[..., 1 : N + 1] - lateness_lambda * lateness
        return logits

    @staticmethod
    def _manhattan(a_xy: torch.Tensor, b_xy: torch.Tensor) -> torch.Tensor:
        return (a_xy[..., 0] - b_xy[..., 0]).abs() + (a_xy[..., 1] - b_xy[..., 1]).abs()

    def forward(
        self,
        *,
        feats: Dict[str, torch.Tensor],
        agents: torch.Tensor,
        k: int,
        lateness_lambda: float = 0.0,
        cap_full: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if k < 1:
            raise ValueError("k must be >= 1")
        nodes = feats["nodes"]
        depot = feats["depot"]
        enc = self.encode(feats)
        H_nodes = enc["H_nodes"]
        H_depot = enc["H_depot"]
        mask = enc["node_mask"].clone()
        B, N, _ = nodes.shape
        A = agents.size(1)
        ag = agents.clone()
        if cap_full is None:
            raise RuntimeError("cap_full must be provided when running DVRPNet.forward")
        cap_full_local = cap_full.clone()
        out_idx = torch.zeros(B, A, k, dtype=torch.long, device=nodes.device)
        out_xy = torch.zeros(B, A, k, 2, dtype=torch.long, device=nodes.device)
        node_xy = nodes[..., :2].long()
        depot_xy = depot.squeeze(1).long()

        for step in range(k):
            logits = self.decode(
                enc_nodes=H_nodes,
                enc_depot=H_depot,
                node_mask=mask,
                agents_tensor=ag,
                nodes=nodes,
                lateness_lambda=lateness_lambda,
                time_now=feats.get("time_now"),
            )
            sel = torch.zeros(B, A, dtype=torch.long, device=nodes.device)
            taken = torch.zeros(B, N, dtype=torch.bool, device=nodes.device) if N > 0 else None
            neg_inf = torch.finfo(logits.dtype).min
            for agent_idx in range(A):
                logits_view = logits[:, agent_idx, :].clone()
                if N > 0 and taken is not None:
                    duplicate_mask = taken
                    logits_view[:, 1 : N + 1] = torch.where(
                        duplicate_mask,
                        torch.full_like(logits_view[:, 1 : N + 1], neg_inf),
                        logits_view[:, 1 : N + 1],
                    )
                sel[:, agent_idx] = torch.argmax(logits_view, dim=-1)
                if N > 0 and taken is not None:
                    for b in range(B):
                        idx = int(sel[b, agent_idx].item())
                        if 1 <= idx <= N:
                            taken[b, idx - 1] = True
            dest_xy = torch.zeros(B, A, 2, dtype=torch.long, device=nodes.device)
            for b in range(B):
                for a in range(A):
                    idx = int(sel[b, a].item())
                    if 1 <= idx <= N:
                        dest_xy[b, a] = node_xy[b, idx - 1]
                    else:
                        dest_xy[b, a] = depot_xy[b]
            out_idx[..., step] = sel
            out_xy[:, :, step, :] = dest_xy
            if N > 0:
                for b in range(B):
                    for a in range(A):
                        idx = int(sel[b, a].item())
                        if 1 <= idx <= N:
                            mask[b, idx - 1] = True
            cur_xy = ag[..., :2].long()
            dist = self._manhattan(cur_xy, dest_xy).to(ag.dtype)
            ag[..., :2] = dest_xy.to(ag.dtype)
            ag[..., 3] = ag[..., 3] + dist
            for b in range(B):
                for a in range(A):
                    idx = int(sel[b, a].item())
                    if 1 <= idx <= N:
                        demand = nodes[b, idx - 1, 3].to(ag.dtype)
                        ag[b, a, 2] = torch.clamp(ag[b, a, 2] - demand, min=0.0)
                    else:
                        ag[b, a, 2] = cap_full_local[b, a]
        return {"indices": out_idx, "coords": out_xy}

    def decode_step(
        self,
        feats: Dict[str, torch.Tensor],
        lateness_lambda: float = 0.0,
        current_time: float | int = 0,
    ) -> torch.Tensor:
        if "agents" not in feats:
            raise ValueError("decode_step requires 'agents' tensor in feats")
        feats = dict(feats)
        feats.setdefault(
            "time_now",
            torch.tensor(current_time, dtype=torch.float32, device=feats["nodes"].device),
        )
        enc = self.encode(feats)
        logits = self.decode(
            enc_nodes=enc["H_nodes"],
            enc_depot=enc["H_depot"],
            node_mask=enc["node_mask"],
            agents_tensor=feats["agents"],
            nodes=feats.get("nodes"),
            lateness_lambda=lateness_lambda,
            time_now=feats.get("time_now"),
        )
        if logits.size(1) == 1:
            return logits.squeeze(1)
        return logits[:, 0, :]

    def freeze_base_and_train_adapters(self) -> None:
        adapters_found = False
        for name, param in self.named_parameters():
            is_adapter = "adapter" in name
            param.requires_grad = is_adapter
            adapters_found |= is_adapter
        if not adapters_found:
            raise RuntimeError(
                "Adapter-only training requested but no adapter parameters were found. "
                "Set adapter_dim > 0 when instantiating DVRPNet for the dynamic stage."
            )

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
