"""
Static VRP Model - POMO-style architecture adapted for multi-vehicle VRP.

Key design decisions:
1. Single vehicle constructs the full tour (visiting all nodes and returning to depot)
2. Tour is split at depot returns to create multi-vehicle routes
3. Output: route assignments for each vehicle

This matches the original POMO paper's approach but with explicit multi-vehicle output.

NORMALIZATION (v2 - capacity-normalized, see configs.py):
- Coordinates: [0, COORD_NORM] => [0, 1] (default COORD_NORM=20, can be varied)
- Demands: [0, DEMAND_NORM] => [0, 1] (DEMAND_NORM=30 = vehicle capacity, FIXED)
- Vehicle capacity: 1.0 (= 30/30, represents full capacity)
- Max demand per node: 5 (= 0.167 of capacity)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn

# Import standardized normalization constants
try:
    from configs import COORD_NORM, DEMAND_NORM
except ImportError:
    # Fallback for standalone usage
    COORD_NORM = 20.0
    DEMAND_NORM = 30.0


@dataclass
class ResetState:
    """State after environment reset."""
    depot_xy: Optional[torch.Tensor] = None   # (batch, 1, 2)
    node_xy: Optional[torch.Tensor] = None    # (batch, problem, 2)
    node_demand: Optional[torch.Tensor] = None  # (batch, problem)


@dataclass
class StepState:
    """State for each decoding step."""
    BATCH_IDX: Optional[torch.Tensor] = None  # (batch, pomo)
    POMO_IDX: Optional[torch.Tensor] = None   # (batch, pomo)
    selected_count: int = 0
    load: Optional[torch.Tensor] = None       # (batch, pomo) - remaining capacity
    current_node: Optional[torch.Tensor] = None  # (batch, pomo)
    ninf_mask: Optional[torch.Tensor] = None  # (batch, pomo, problem+1)
    finished: Optional[torch.Tensor] = None   # (batch, pomo)


class StaticVRPEnv:
    """
    Static VRP Environment - POMO style.
    
    Generates a single tour that visits all nodes, returning to depot when
    capacity is exhausted. The tour can be split into multi-vehicle routes.
    """

    def __init__(
        self,
        problem_size: int = 100,
        pomo_size: int = 100,
        vehicle_capacity: float = 1.0,
    ):
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.vehicle_capacity = vehicle_capacity

        # Batch state
        self.batch_size: int = 0
        self.BATCH_IDX: Optional[torch.Tensor] = None
        self.POMO_IDX: Optional[torch.Tensor] = None
        
        # Problem data
        self.depot_node_xy: Optional[torch.Tensor] = None  # (batch, problem+1, 2)
        self.depot_node_demand: Optional[torch.Tensor] = None  # (batch, problem+1)
        
        # Dynamic state
        self.selected_count: int = 0
        self.current_node: Optional[torch.Tensor] = None
        self.selected_node_list: Optional[torch.Tensor] = None
        self.at_the_depot: Optional[torch.Tensor] = None
        self.load: Optional[torch.Tensor] = None
        self.visited_ninf_flag: Optional[torch.Tensor] = None
        self.ninf_mask: Optional[torch.Tensor] = None
        self.finished: Optional[torch.Tensor] = None
        
        # States to return
        self.reset_state = ResetState()
        self.step_state = StepState()

    def load_problems(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        aug_factor: int = 1,
    ):
        """
        Load problem instances.
        
        Args:
            depot_xy: (batch, 1, 2) depot coordinates (normalized to [0,1])
            node_xy: (batch, problem, 2) node coordinates (normalized to [0,1])
            node_demand: (batch, problem) node demands (normalized so total <= capacity)
            aug_factor: data augmentation factor (1 or 8)
        """
        self.batch_size = depot_xy.size(0)
        device = depot_xy.device

        if aug_factor == 8:
            self.batch_size *= 8
            depot_xy = self._augment_xy_8fold(depot_xy)
            node_xy = self._augment_xy_8fold(node_xy)
            node_demand = node_demand.repeat(8, 1)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(self.batch_size, 1, device=device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(
            self.batch_size, self.pomo_size
        )

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def load_problems_from_raw(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        coord_scale: float = COORD_NORM,
        capacity: float = DEMAND_NORM,
    ):
        """
        Load problems from raw (unnormalized) coordinates and demands.
        
        Args:
            depot_xy: (batch, 1, 2) or (batch, 2) depot coordinates
            node_xy: (batch, problem, 2) node coordinates
            node_demand: (batch, problem) node demands
            coord_scale: max coordinate value for normalization (default: COORD_NORM)
            capacity: demand normalization constant (default: DEMAND_NORM = 30)
        """
        device = depot_xy.device
        
        # Normalize coordinates to [0, 1]
        if depot_xy.dim() == 2:
            depot_xy = depot_xy.unsqueeze(1)
        depot_norm = depot_xy / coord_scale
        node_norm = node_xy / coord_scale
        
        # Normalize demands (demand / capacity)
        demand_norm = node_demand / capacity
        
        self.load_problems(depot_norm, node_norm, demand_norm)

    def reset(self) -> Tuple[ResetState, None, bool]:
        """Reset environment to initial state."""
        device = self.depot_node_xy.device
        
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long, device=device
        )
        
        self.at_the_depot = torch.ones(
            (self.batch_size, self.pomo_size), dtype=torch.bool, device=device
        )
        self.load = torch.ones(
            (self.batch_size, self.pomo_size), device=device
        ) * self.vehicle_capacity
        self.visited_ninf_flag = torch.zeros(
            (self.batch_size, self.pomo_size, self.problem_size + 1), device=device
        )
        self.ninf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, self.problem_size + 1), device=device
        )
        self.finished = torch.zeros(
            (self.batch_size, self.pomo_size), dtype=torch.bool, device=device
        )
        
        return self.reset_state, None, False

    def pre_step(self) -> Tuple[StepState, None, bool]:
        """Prepare state for first step."""
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        return self.step_state, None, False

    def step(self, selected: torch.Tensor) -> Tuple[StepState, Optional[torch.Tensor], bool]:
        """
        Execute one step.
        
        Args:
            selected: (batch, pomo) selected node indices (0=depot, 1..N=nodes)
            
        Returns:
            state, reward, done
        """
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )
        
        # Update depot status
        self.at_the_depot = (selected == 0)
        
        # Update load
        demand_list = self.depot_node_demand[:, None, :].expand(
            self.batch_size, self.pomo_size, -1
        )
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load = self.load - selected_demand
        self.load[self.at_the_depot] = self.vehicle_capacity  # refill at depot
        
        # Update visited mask
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # Allow depot revisit unless at depot
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0
        
        # Update action mask (visited + capacity constraint)
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')
        
        # Check termination
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished | newly_finished
        
        # Allow depot for finished episodes
        self.ninf_mask[:, :, 0][self.finished] = 0
        
        # Update step state
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        
        done = self.finished.all()
        reward = -self._get_travel_distance() if done else None
        
        return self.step_state, reward, done

    def _get_travel_distance(self) -> torch.Tensor:
        """Calculate total travel distance for all tours."""
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        return segment_lengths.sum(2)

    def _augment_xy_8fold(self, xy_data: torch.Tensor) -> torch.Tensor:
        """8-fold data augmentation by rotation and reflection."""
        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        
        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)
        
        return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)

    def get_multi_vehicle_routes(
        self,
        selected_list: torch.Tensor,
        num_vehicles: int,
    ) -> List[List[List[int]]]:
        """
        Convert single tour to multi-vehicle routes by splitting at depot returns.
        
        Args:
            selected_list: (batch, pomo, seq_len) selected node sequence
            num_vehicles: maximum number of vehicles
            
        Returns:
            List of routes for each batch, each pomo instance
        """
        batch_size, pomo_size, seq_len = selected_list.shape
        all_routes = []
        
        for b in range(batch_size):
            batch_routes = []
            for p in range(pomo_size):
                tour = selected_list[b, p].tolist()
                routes = []
                current_route = []
                
                for node in tour:
                    if node == 0:  # depot
                        if current_route:
                            routes.append(current_route)
                            current_route = []
                    else:
                        current_route.append(node)
                
                if current_route:
                    routes.append(current_route)
                
                # Pad or truncate to num_vehicles
                while len(routes) < num_vehicles:
                    routes.append([])
                routes = routes[:num_vehicles]
                
                batch_routes.append(routes)
            all_routes.append(batch_routes)
        
        return all_routes


class Encoder(nn.Module):
    """POMO-style encoder with multi-head attention."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        encoder_layer_num: int = 6,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
    ):
        super().__init__()
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)  # (x, y, demand)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
            for _ in range(encoder_layer_num)
        ])

    def forward(
        self,
        depot_xy: torch.Tensor,
        node_xy_demand: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depot_xy: (batch, 1, 2)
            node_xy_demand: (batch, problem, 3) - (x, y, demand)
            
        Returns:
            encoded: (batch, problem+1, embedding_dim)
        """
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        
        for layer in self.layers:
            out = layer(out)
        
        return out


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feedforward."""
    
    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        ff_hidden_dim: int,
    ):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q = self._reshape_by_heads(self.Wq(x))
        k = self._reshape_by_heads(self.Wk(x))
        v = self._reshape_by_heads(self.Wv(x))
        
        attn = self._multi_head_attention(q, k, v)
        mh_out = self.multi_head_combine(attn)
        
        # Add & Norm
        out1 = self.norm1(x + mh_out)
        
        # Feedforward & Add & Norm
        out2 = self.norm2(out1 + self.ff(out1))
        
        return out2

    def _reshape_by_heads(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, n, _ = qkv.shape
        return qkv.view(batch_size, n, self.head_num, self.qkv_dim).transpose(1, 2)

    def _multi_head_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        weights = torch.softmax(score, dim=-1)
        out = torch.matmul(weights, v)
        return out.transpose(1, 2).reshape(q.size(0), q.size(2), -1)


class Decoder(nn.Module):
    """POMO-style decoder with context attention."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        head_num: int = 8,
        qkv_dim: int = 16,
        logit_clipping: float = 10.0,
    ):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.sqrt_embedding_dim = math.sqrt(embedding_dim)
        self.logit_clipping = logit_clipping
        
        # Query projection: last_node_emb + load
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        # Cached keys and values
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.single_head_key: Optional[torch.Tensor] = None

    def set_kv(self, encoded_nodes: torch.Tensor):
        """Cache keys and values from encoded nodes."""
        self.k = self._reshape_by_heads(self.Wk(encoded_nodes))
        self.v = self._reshape_by_heads(self.Wv(encoded_nodes))
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(
        self,
        encoded_last_node: torch.Tensor,
        load: torch.Tensor,
        ninf_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoded_last_node: (batch, pomo, embedding_dim)
            load: (batch, pomo) remaining capacity ratio
            ninf_mask: (batch, pomo, problem+1)
            
        Returns:
            probs: (batch, pomo, problem+1) action probabilities
            selected: (batch, pomo) selected nodes
        """
        # Concatenate last node embedding with load
        input_cat = torch.cat((encoded_last_node, load.unsqueeze(-1)), dim=-1)
        
        # Attention
        q_last = self._reshape_by_heads(self.Wq_last(input_cat))
        attn = self._multi_head_attention(q_last, self.k, self.v, ninf_mask)
        mh_out = self.multi_head_combine(attn)
        
        # Score calculation
        score = torch.matmul(mh_out, self.single_head_key) / self.sqrt_embedding_dim
        score = self.logit_clipping * torch.tanh(score)
        score = score + ninf_mask
        
        probs = torch.softmax(score, dim=-1)
        
        # Sample or argmax based on training mode
        if self.training:
            selected = probs.reshape(-1, probs.size(-1)).multinomial(1).reshape(
                probs.size(0), probs.size(1)
            )
        else:
            selected = probs.argmax(dim=-1)
        
        # Get selected probability
        prob = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        
        return selected, prob

    def _reshape_by_heads(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, n, _ = qkv.shape
        return qkv.view(batch_size, n, self.head_num, self.qkv_dim).transpose(1, 2)

    def _multi_head_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ninf_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        if ninf_mask is not None:
            score = score + ninf_mask[:, None, :, :]
        weights = torch.softmax(score, dim=-1)
        out = torch.matmul(weights, v)
        return out.transpose(1, 2).reshape(q.size(0), q.size(2), -1)


class StaticVRPModel(nn.Module):
    """
    POMO-style model for static VRP.
    
    Single vehicle constructs tour visiting all nodes, which can be
    split into multi-vehicle routes at depot returns.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        encoder_layer_num: int = 6,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
        logit_clipping: float = 10.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            encoder_layer_num=encoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            head_num=head_num,
            qkv_dim=qkv_dim,
            logit_clipping=logit_clipping,
        )
        
        self.encoded_nodes: Optional[torch.Tensor] = None

    def pre_forward(self, reset_state: ResetState):
        """Encode problem instance."""
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        
        # Combine xy and demand for nodes
        node_xy_demand = torch.cat((node_xy, node_demand.unsqueeze(-1)), dim=-1)
        
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state: StepState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode one step.
        
        Args:
            state: current step state
            
        Returns:
            selected: (batch, pomo) selected node indices
            prob: (batch, pomo) selection probabilities
        """
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.selected_count == 0:
            # First step: all start from depot
            selected = torch.zeros(batch_size, pomo_size, dtype=torch.long,
                                   device=state.BATCH_IDX.device)
            prob = torch.ones(batch_size, pomo_size, device=state.BATCH_IDX.device)
        elif state.selected_count == 1:
            # Second step: POMO - each pomo instance starts from different node
            # Select top-pomo_size nodes for diverse starting points
            encoded_depot = self.encoded_nodes[:, [0], :]  # (batch, 1, emb)
            encoded_depot = encoded_depot.expand(-1, pomo_size, -1)
            load = state.load
            
            selected, prob = self.decoder(encoded_depot, load, state.ninf_mask)
            
            # For POMO: assign different first nodes to different pomo instances
            if pomo_size > 1 and batch_size == 1:
                # Greedy diverse selection for single batch
                probs_flat = torch.softmax(
                    self.decoder.multi_head_combine(
                        self.decoder._multi_head_attention(
                            self.decoder._reshape_by_heads(
                                self.decoder.Wq_last(
                                    torch.cat((encoded_depot, load.unsqueeze(-1)), dim=-1)
                                )
                            ),
                            self.decoder.k,
                            self.decoder.v,
                            state.ninf_mask,
                        )
                    ).matmul(self.decoder.single_head_key) / self.decoder.sqrt_embedding_dim
                    + state.ninf_mask,
                    dim=-1,
                )
                # Select top-k different nodes
                _, top_indices = probs_flat[0, 0].topk(min(pomo_size, probs_flat.size(-1) - 1))
                if top_indices.size(0) < pomo_size:
                    top_indices = torch.cat([
                        top_indices,
                        top_indices.new_ones(pomo_size - top_indices.size(0))
                    ])
                selected = top_indices[:pomo_size].unsqueeze(0)
                prob = probs_flat.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        else:
            # Subsequent steps: decode based on current node
            gathering_index = state.current_node[:, :, None].expand(-1, -1, self.embedding_dim)
            encoded_last_node = self.encoded_nodes[:, None, :, :].expand(
                -1, pomo_size, -1, -1
            ).gather(2, gathering_index.unsqueeze(2)).squeeze(2)
            
            selected, prob = self.decoder(encoded_last_node, state.load, state.ninf_mask)
        
        return selected, prob

    def solve(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        pomo_size: int = 8,
        num_vehicles: int = 2,
        augment: bool = False,
        vehicle_capacity: float = 1.0,
    ) -> Tuple[torch.Tensor, List[List[List[int]]]]:
        """
        Solve VRP instance(s).
        
        Args:
            depot_xy: (batch, 1, 2) or (batch, 2) - normalized depot coordinates
            node_xy: (batch, problem, 2) - normalized node coordinates
            node_demand: (batch, problem) - normalized demands (demand / demand_norm)
            pomo_size: number of parallel rollouts
            num_vehicles: number of vehicles for route splitting
            augment: use 8-fold data augmentation
            vehicle_capacity: normalized vehicle capacity (capacity / demand_norm)
            
        Returns:
            distances: (batch,) best tour distances
            routes: multi-vehicle routes for each instance
        """
        if depot_xy.dim() == 2:
            depot_xy = depot_xy.unsqueeze(1)
        
        batch_size = depot_xy.size(0)
        problem_size = node_xy.size(1)
        
        env = StaticVRPEnv(problem_size=problem_size, pomo_size=pomo_size, vehicle_capacity=vehicle_capacity)
        aug_factor = 8 if augment else 1
        env.load_problems(depot_xy, node_xy, node_demand, aug_factor=aug_factor)
        
        self.eval()
        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.pre_forward(reset_state)
            
            state, _, done = env.pre_step()
            while not done:
                selected, _ = self(state)
                state, reward, done = env.step(selected)
        
        # Get best solution from POMO
        if augment:
            reward = reward.reshape(8, batch_size, pomo_size)
            best_pomo_reward, best_pomo_idx = reward.max(dim=2)
            best_aug_reward, best_aug_idx = best_pomo_reward.max(dim=0)
            
            # Extract best routes
            selected_list = env.selected_node_list.reshape(8, batch_size, pomo_size, -1)
            best_routes_list = []
            for b in range(batch_size):
                aug_i = best_aug_idx[b].item()
                pomo_i = best_pomo_idx[aug_i, b].item()
                tour = selected_list[aug_i, b, pomo_i]
                best_routes_list.append(tour)
            
            distances = -best_aug_reward
        else:
            best_reward, best_idx = reward.max(dim=1)
            distances = -best_reward
            
            best_routes_list = []
            for b in range(batch_size):
                tour = env.selected_node_list[b, best_idx[b].item()]
                best_routes_list.append(tour)
        
        # Convert to multi-vehicle routes - simple segment split for shortest path
        routes = []
        for b_idx, tour in enumerate(best_routes_list):
            vehicle_routes = self._split_tour_to_routes(tour.tolist(), num_vehicles)
            routes.append(vehicle_routes)
        
        return distances, routes

    def _split_tour_to_routes(
        self,
        tour: List[int],
        num_vehicles: int,
    ) -> List[List[int]]:
        """Split single tour into multi-vehicle routes at depot returns.
        
        Since all agents start from depot and return to depot, and the optimal
        tour also forms cycles (depot -> nodes -> depot) due to capacity limits,
        we should NEVER break a cycle. Instead:
        
        1. If num_cycles <= num_vehicles: assign one cycle per vehicle
        2. If num_cycles > num_vehicles: some vehicles do multiple cycles sequentially
        
        Args:
            tour: List of node indices (0 = depot)
            num_vehicles: Number of vehicles
            
        Returns:
            List of routes, one per vehicle. Each route is a sequence of nodes
            ending with depot (0). If a vehicle does multiple cycles, they are
            concatenated (node1, node2, 0, node3, node4, 0 means two cycles).
        """
        # Extract cycles (complete routes: depot -> nodes -> depot)
        # Each cycle is a list of customer nodes (excluding depot)
        cycles = []
        current_cycle = []
        
        for node in tour:
            if node == 0:  # depot
                if current_cycle:
                    cycles.append(current_cycle)
                    current_cycle = []
            else:
                current_cycle.append(node)
        
        # Don't forget the last cycle if tour doesn't end with depot
        if current_cycle:
            cycles.append(current_cycle)
        
        if not cycles:
            # No nodes to visit, each vehicle just stays at depot
            return [[0] for _ in range(num_vehicles)]
        
        # Initialize routes for each vehicle
        routes = [[] for _ in range(num_vehicles)]
        
        if len(cycles) <= num_vehicles:
            # Case 1: Fewer or equal cycles than vehicles
            # Assign one cycle per vehicle (some vehicles may be idle)
            for i, cycle in enumerate(cycles):
                routes[i].extend(cycle)
                routes[i].append(0)  # Return to depot
            # Idle vehicles just have depot
            for i in range(len(cycles), num_vehicles):
                routes[i].append(0)
        else:
            # Case 2: More cycles than vehicles
            # Distribute cycles to vehicles, keeping cycles intact
            # Use balanced assignment: assign to vehicle with least total nodes
            loads = [0] * num_vehicles
            
            for cycle in cycles:
                # Find vehicle with minimum load
                min_vehicle = min(range(num_vehicles), key=lambda v: loads[v])
                # Append the complete cycle (nodes + depot return)
                routes[min_vehicle].extend(cycle)
                routes[min_vehicle].append(0)  # Depot marks end of this cycle
                loads[min_vehicle] += len(cycle)
        
        return routes


def create_static_model(
    embedding_dim: int = 128,
    encoder_layers: int = 6,
    heads: int = 8,
    qkv_dim: int = 16,
    ff_hidden: int = 512,
) -> StaticVRPModel:
    """Factory function to create static VRP model."""
    return StaticVRPModel(
        embedding_dim=embedding_dim,
        encoder_layer_num=encoder_layers,
        head_num=heads,
        qkv_dim=qkv_dim,
        ff_hidden_dim=ff_hidden,
    )
