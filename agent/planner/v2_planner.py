"""
V2 Planner - Uses the new models_v2 architecture (Static VRP + Dynamic Adapter).

This planner replaces the old ModelPlanner and CVRPPOMOPlanner with the unified
POMO-based static model + residual adapter architecture.

NORMALIZATION SCHEME (v2 - capacity-normalized):
------------------------------------------------
1. Coordinates: Normalized to [0,1] by dividing by COORD_NORM (grid size)
2. Demands: Normalized by DEMAND_NORM (= vehicle capacity = 30)
3. Vehicle Capacity: vehicle_capacity = capacity / DEMAND_NORM = 30/30 = 1.0

Example:
- DEMAND_NORM = 30 (= vehicle capacity)
- capacity = 30 (fixed)
- demand = 5 (per node)
=> normalized_demand = 5/30 = 0.167
=> vehicle_capacity = 30/30 = 1.0
=> Model sees 0.167 demand vs 1.0 capacity
"""
from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Dict, Any
from collections import deque
import os

from .base import BasePlanner, AgentState, Target

import torch

# Import normalization constants
from configs import COORD_NORM, DEMAND_NORM


class V2Planner(BasePlanner):
    """
    使用 models_v2 架构的 Planner。
    
    架构：
    - StaticVRPModel: POMO 风格的静态 VRP 模型（冻结）
    - DynamicVRPModel: 静态模型 + 残差适配器（可训练）
    
    支持两种模式：
    - static: 仅使用静态模型（用于静态 VRP 问题）
    - dynamic: 使用静态模型 + 适配器（用于动态 VRP 问题）
    
    Normalization is handled automatically using constants from configs.py:
    - COORD_NORM: Grid size for coordinate normalization (default 20)
    - DEMAND_NORM: Vehicle capacity normalization constant (= 30)
    """

    def __init__(
        self,
        mode: str = "dynamic",  # "static" or "dynamic"
        static_checkpoint: Optional[str] = None,
        adapter_checkpoint: Optional[str] = None,
        embedding_dim: int = 128,
        encoder_layers: int = 6,
        heads: int = 8,
        qkv_dim: int = 16,
        ff_hidden: int = 512,
        adapter_dim: int = 32,
        device: str = "cuda",
        grid_width: int = 20,
        grid_height: int = 20,
        full_capacity: int = 30,  # Fixed vehicle capacity (= DEMAND_NORM)
        max_time: int = 100,
        pomo_size: int = 20,
        aug_factor: int = 8,
        **params,
    ) -> None:
        super().__init__(**params)
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.heads = heads
        self.qkv_dim = qkv_dim
        self.ff_hidden = ff_hidden
        self.adapter_dim = adapter_dim
        self.device = device
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.full_capacity = full_capacity
        self.max_time = max_time
        self.pomo_size = pomo_size
        self.aug_factor = aug_factor
        
        # Normalization constants (use standardized values from configs)
        # coord_norm: normalize coordinates to [0,1] range
        self.coord_norm = COORD_NORM
        # demand_norm: FIXED constant = vehicle capacity = 30
        # This ensures demands and capacity are normalized consistently
        self.demand_norm = DEMAND_NORM
        # capacity_norm: the actual vehicle capacity (for load display/tracking)
        self.capacity_norm = float(full_capacity)
        self.time_norm = float(max_time)
        
        # Vehicle capacity for the model = full_capacity / demand_norm
        # With full_capacity=30 and demand_norm=30, this equals 1.0
        self._model_vehicle_capacity = float(full_capacity) / self.demand_norm
        
        self._model = None
        self._static_checkpoint = static_checkpoint
        self._adapter_checkpoint = adapter_checkpoint
        
        # Lazy load model
        self._loaded = False

    def _ensure_model_loaded(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
            
        from models_v2.static_model import create_static_model
        from models_v2.dynamic_model import create_dynamic_model
        
        if self.mode == "static":
            # Static mode: only use static model
            self._model = create_static_model(
                embedding_dim=self.embedding_dim,
                encoder_layers=self.encoder_layers,
                heads=self.heads,
                qkv_dim=self.qkv_dim,
                ff_hidden=self.ff_hidden,
            )
            if self._static_checkpoint and os.path.exists(self._static_checkpoint):
                ckpt = torch.load(self._static_checkpoint, map_location=self.device)
                if 'model_state_dict' in ckpt:
                    self._model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self._model.load_state_dict(ckpt)
                print(f"[V2Planner] Loaded static model from {self._static_checkpoint}")
            self._model = self._model.to(self.device)
            self._model.eval()
        else:
            # Dynamic mode: static model + adapter
            self._model = create_dynamic_model(
                static_model_or_checkpoint=self._static_checkpoint,
                embedding_dim=self.embedding_dim,
                encoder_layers=self.encoder_layers,
                heads=self.heads,
                qkv_dim=self.qkv_dim,
                ff_hidden=self.ff_hidden,
                adapter_dim=self.adapter_dim,
                freeze_static=True,
                device=self.device,
            )
            if self._adapter_checkpoint and os.path.exists(self._adapter_checkpoint):
                ckpt = torch.load(self._adapter_checkpoint, map_location=self.device)
                if 'adapter_state' in ckpt:
                    self._model.load_adapter_state_dict(ckpt['adapter_state'])
                elif 'adapter_state_dict' in ckpt:
                    self._model.load_adapter_state_dict(ckpt['adapter_state_dict'])
                print(f"[V2Planner] Loaded adapter from {self._adapter_checkpoint}")
            self._model.eval()
        
        self._loaded = True

    def load_from_ckpt(self, ckpt_path: str) -> None:
        """Load checkpoint (for compatibility with old API)."""
        # Determine if this is a static or adapter checkpoint
        if 'adapter' in ckpt_path.lower():
            self._adapter_checkpoint = ckpt_path
        else:
            self._static_checkpoint = ckpt_path
        self._loaded = False  # Force reload

    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],  # [(x,y,t_arrival,demand,t_due), ...]
        agent_states: List[AgentState],  # x,y,s (load)
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        """
        返回每个 agent 的目标队列（deque[(x,y), ...]）
        
        Args:
            observations: 当前可见的需求列表 [(x, y, t_arrival, capacity, t_due), ...]
            agent_states: agent 状态列表 [AgentState(x, y, s), ...]
            depot: depot 坐标 (x, y)
            t: 当前时间
            horizon: 规划时间窗口
            current_plans: 当前已有的规划路径（用于延续执行）
            
        Returns:
            每个 agent 的目标队列
        """
        self._ensure_model_loaded()
        
        num_agents = len(agent_states)
        nodes = list(observations)
        N = len(nodes)
        
        # Static mode: plan only once, then follow the existing plan
        # This avoids re-planning every step which causes route oscillation
        if self.mode == "static" and current_plans is not None:
            # Check if we have valid existing plans with remaining targets
            has_valid_plans = any(len(plan) > 0 for plan in current_plans)
            if has_valid_plans:
                # Return the SAME deque references so that controller's popleft() persists
                # This is critical: returning copies would reset the pop effect each step
                return list(current_plans)
        
        # If no nodes, all agents return to depot
        if N == 0:
            return [deque([depot] * max(1, horizon)) for _ in range(num_agents)]
        
        with torch.no_grad():
            # Prepare input tensors
            # Depot: (1, 1, 2)
            depot_xy = torch.tensor([[list(depot)]], dtype=torch.float32, device=self.device)
            depot_xy = depot_xy / self.coord_norm
            
            # Nodes: (1, N, 2) and demands: (1, N)
            node_coords = [[n[0], n[1]] for n in nodes]
            node_demands = [n[3] for n in nodes]  # capacity/demand
            
            node_xy = torch.tensor([node_coords], dtype=torch.float32, device=self.device)
            node_xy = node_xy / self.coord_norm
            
            node_demand = torch.tensor([node_demands], dtype=torch.float32, device=self.device)
            # Use demand_norm (fixed training constant) for demand normalization
            # This makes capacity an adjustable absolute value
            node_demand = node_demand / self.demand_norm
            
            # Agent states: (1, A, 4) - [x, y, load, time]
            agent_data = []
            for a in agent_states:
                agent_data.append([
                    a.x / self.coord_norm,
                    a.y / self.coord_norm,
                    a.s / self.capacity_norm,  # current load
                    t / self.time_norm,
                ])
            agent_states_tensor = torch.tensor([agent_data], dtype=torch.float32, device=self.device)
            
            # Mask: (1, A, N+1) - depot is index 0, nodes are 1..N
            ninf_mask = torch.zeros(1, num_agents, N + 1, device=self.device)
            
            if self.mode == "static":
                # Static model uses solve() method with configurable POMO params
                effective_pomo = min(N, self.pomo_size)
                use_augment = (self.aug_factor == 8)
                # Use pre-computed model vehicle capacity (full_capacity / DEMAND_NORM_TRAINING)
                distances, routes = self._model.solve(
                    depot_xy, node_xy, node_demand,
                    pomo_size=effective_pomo,
                    num_vehicles=num_agents,
                    augment=use_augment,
                    vehicle_capacity=self._model_vehicle_capacity,
                )
                # Convert routes to target queues
                # Routes from model already include depot (0) at the end
                result = []
                for a in range(num_agents):
                    targets = deque()
                    if a < len(routes[0]):
                        for node_idx in routes[0][a]:
                            if node_idx == 0:
                                targets.append(depot)
                            else:
                                idx = node_idx - 1
                                if idx < len(nodes):
                                    targets.append((nodes[idx][0], nodes[idx][1]))
                    if not targets:
                        targets.append(depot)
                    result.append(targets)
                return result
            else:
                # Dynamic model generates complete tour then distributes segments
                # Goal: shortest total distance, same as static model
                
                # Generate complete tour using greedy sequential decoding
                full_tour = self._generate_full_tour_dynamic(
                    depot_xy, node_xy, node_demand,
                    agent_states_tensor, ninf_mask, N,
                )
                
                # Split tour into segments at depot returns
                segments = self._extract_segments(full_tour)
                
                # Simple sequential distribution (no balancing for shortest path)
                vehicle_segments = self._distribute_segments_sequential(
                    segments, num_agents,
                )
                
                # Convert to target queues
                # Routes from _distribute_segments_sequential already include depot (0) at the end
                result = []
                for a in range(num_agents):
                    targets = deque()
                    if a < len(vehicle_segments):
                        for node_idx in vehicle_segments[a]:
                            if node_idx == 0:
                                targets.append(depot)
                            else:
                                idx = node_idx - 1
                                if idx < len(nodes):
                                    targets.append((nodes[idx][0], nodes[idx][1]))
                    if not targets:
                        targets.append(depot)
                    result.append(targets)
                
                return result
    
    def _generate_full_tour_dynamic(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        agent_states: torch.Tensor,
        ninf_mask: torch.Tensor,
        num_nodes: int,
    ) -> List[int]:
        """
        Generate a complete tour visiting all nodes using dynamic model.
        
        Returns a tour as list of node indices (0 = depot, 1..N = nodes).
        Tour visits all nodes, returning to depot when capacity is exhausted.
        """
        tour = []
        visited = set()
        current_load = 0.0
        capacity = 1.0  # Normalized capacity
        
        # Create a working copy of states and masks
        working_mask = ninf_mask.clone()
        working_states = agent_states.clone()
        
        max_steps = num_nodes * 2 + 5  # Allow for depot returns
        
        for step in range(max_steps):
            if len(visited) >= num_nodes:
                break
            
            # Update mask: visited nodes should be blocked
            for v in visited:
                working_mask[0, :, v + 1] = float('-inf')  # v is 0-indexed, mask uses 1-indexed
            
            # Get probabilities from model for all agents
            try:
                _, probs = self._model.forward_with_full_probs(
                    depot_xy, node_xy, node_demand,
                    working_states, working_mask,
                )
            except Exception as e:
                # If model fails, break early
                print(f"[DEBUG] Model forward failed: {e}")
                break
            
            # Use agent 0's probability for building the single tour
            agent_probs = probs[0, 0].clone()  # (N+1,)
            
            # Additional mask for visited nodes (in case forward doesn't apply mask)
            for v in visited:
                agent_probs[v + 1] = float('-inf')
            
            # Handle case where all nodes are visited
            valid_mask = agent_probs > float('-inf')
            if not valid_mask.any():
                break
            
            # Get the selected node
            selected = agent_probs.argmax().item()
            
            if selected == 0:
                # Return to depot (reset load)
                if tour and tour[-1] != 0:
                    tour.append(0)
                current_load = 0.0
            else:
                node_idx = selected - 1  # Convert to 0-indexed
                if node_idx < num_nodes and node_idx not in visited:
                    demand = node_demand[0, node_idx].item()
                    
                    # Check capacity constraint
                    if current_load + demand > capacity:
                        # Return to depot first
                        if tour and tour[-1] != 0:
                            tour.append(0)
                        current_load = 0.0
                    
                    tour.append(selected)  # 1-indexed in tour
                    visited.add(node_idx)
                    current_load += demand
                else:
                    # Already visited or invalid index, try depot or another node
                    if not tour or tour[-1] != 0:
                        tour.append(0)
                    current_load = 0.0
        
        # Add any remaining unvisited nodes (fallback)
        remaining = set(range(num_nodes)) - visited
        for node_idx in remaining:
            demand = node_demand[0, node_idx].item()
            if current_load + demand > capacity:
                tour.append(0)
                current_load = 0.0
            tour.append(node_idx + 1)  # 1-indexed
            current_load += demand
        
        # Ensure we end at depot
        if tour and tour[-1] != 0:
            tour.append(0)
        
        return tour
    
    def _extract_segments(self, tour: List[int]) -> List[List[int]]:
        """
        Extract segments from tour (routes between depot visits).
        
        Args:
            tour: List of node indices (0 = depot, 1..N = nodes)
            
        Returns:
            List of segments, each is a list of node indices
        """
        segments = []
        current_segment = []
        
        for node in tour:
            if node == 0:  # depot
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(node)
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _distribute_segments_sequential(
        self,
        segments: List[List[int]],
        num_vehicles: int,
    ) -> List[List[int]]:
        """
        Distribute segments to vehicles with balanced assignment.
        
        Ensures all vehicles participate in serving demands.
        Uses load-balanced distribution to improve completion.
        Each returned route ends with depot (0) for return.
        
        Args:
            segments: List of segments (each segment is list of 1-indexed node indices)
            num_vehicles: Number of vehicles
            
        Returns:
            List of routes (one per vehicle), each route ends with depot (0)
        """
        if not segments:
            # No nodes to visit, each vehicle just has depot
            return [[0] for _ in range(num_vehicles)]
        
        # If we have fewer segments than vehicles, split further
        if len(segments) < num_vehicles and len(segments) > 0:
            # Flatten all nodes and redistribute evenly
            all_nodes = [node for seg in segments for node in seg]
            if len(all_nodes) >= num_vehicles:
                routes = [[] for _ in range(num_vehicles)]
                for i, node in enumerate(all_nodes):
                    vehicle_idx = i % num_vehicles
                    routes[vehicle_idx].append(node)
                # Add depot return to each route
                for route in routes:
                    route.append(0)
                return routes
        
        # Balanced assignment: assign segments to the vehicle with least load
        routes = [[] for _ in range(num_vehicles)]
        loads = [0] * num_vehicles  # Track number of nodes per vehicle
        
        for seg in segments:
            # Find vehicle with minimum load
            min_vehicle = min(range(num_vehicles), key=lambda v: loads[v])
            routes[min_vehicle].extend(seg)
            loads[min_vehicle] += len(seg)
        
        # Add depot return to each route (including empty ones)
        for route in routes:
            route.append(0)
        
        return routes


def create_v2_planner(
    mode: str = "dynamic",
    static_checkpoint: Optional[str] = None,
    adapter_checkpoint: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> V2Planner:
    """Factory function to create V2Planner with default paths."""
    # Default checkpoint paths
    if static_checkpoint is None:
        default_static = "checkpoints/static_vrp_v2/best_n20.pt"
        if os.path.exists(default_static):
            static_checkpoint = default_static
    
    if adapter_checkpoint is None and mode == "dynamic":
        default_adapter = "checkpoints/dynamic_adapter_v2/best_adapter.pt"
        if os.path.exists(default_adapter):
            adapter_checkpoint = default_adapter
    
    return V2Planner(
        mode=mode,
        static_checkpoint=static_checkpoint,
        adapter_checkpoint=adapter_checkpoint,
        device=device,
        **kwargs,
    )
