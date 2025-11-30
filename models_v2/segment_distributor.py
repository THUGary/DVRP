"""
Segment Distributor - Intelligently distribute route segments to multiple vehicles.

The static model generates a single tour visiting all nodes, returning to depot
when capacity is exhausted. This creates segments: depot-demands-depot.

This module handles the intelligent distribution of these segments to vehicles
using various strategies:
1. Round-robin: Simple alternating distribution
2. Load-balanced: Minimize maximum workload difference
3. Distance-balanced: Consider agent starting positions
4. Hungarian: Optimal assignment using Hungarian algorithm
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class Segment:
    """A route segment from depot to depot."""
    nodes: List[int]  # Node indices (1-indexed, 0 is depot)
    total_demand: float
    total_distance: float
    start_pos: Tuple[float, float]  # Starting position (depot)
    end_pos: Tuple[float, float]  # Ending position (depot)
    node_positions: List[Tuple[float, float]]  # Positions of nodes in segment


@dataclass
class AgentInfo:
    """Information about an agent/vehicle."""
    idx: int
    position: Tuple[float, float]
    current_load: float
    capacity: float


def extract_segments_from_tour(
    tour: List[int],
    node_coords: torch.Tensor,  # (N, 2) or (N+1, 2) with depot at 0
    node_demands: torch.Tensor,  # (N,) or (N+1,) with depot at 0
    depot_coord: Tuple[float, float],
) -> List[Segment]:
    """
    Extract segments from a single tour.
    
    A segment is a sequence of nodes visited between two depot visits.
    
    Args:
        tour: List of node indices (0 = depot)
        node_coords: Node coordinates
        node_demands: Node demands
        depot_coord: Depot coordinates
        
    Returns:
        List of Segment objects
    """
    segments = []
    current_nodes = []
    current_positions = []
    current_demand = 0.0
    current_distance = 0.0
    last_pos = depot_coord
    
    for node_idx in tour:
        if node_idx == 0:  # Depot
            if current_nodes:
                # Add distance back to depot
                if current_positions:
                    current_distance += _euclidean_dist(current_positions[-1], depot_coord)
                
                segments.append(Segment(
                    nodes=current_nodes.copy(),
                    total_demand=current_demand,
                    total_distance=current_distance,
                    start_pos=depot_coord,
                    end_pos=depot_coord,
                    node_positions=current_positions.copy(),
                ))
                current_nodes = []
                current_positions = []
                current_demand = 0.0
                current_distance = 0.0
            last_pos = depot_coord
        else:
            # Get node position and demand
            # Handle both (N,) and (N+1,) indexing
            if node_coords.size(0) == node_demands.size(0) + 1:
                # depot at 0, nodes at 1..N
                pos = (node_coords[node_idx, 0].item(), node_coords[node_idx, 1].item())
                demand = node_demands[node_idx - 1].item() if node_idx > 0 else 0.0
            else:
                # nodes only, 0-indexed
                actual_idx = node_idx - 1 if node_idx > 0 else 0
                if actual_idx < node_coords.size(0):
                    pos = (node_coords[actual_idx, 0].item(), node_coords[actual_idx, 1].item())
                    demand = node_demands[actual_idx].item()
                else:
                    continue
            
            current_distance += _euclidean_dist(last_pos, pos)
            current_nodes.append(node_idx)
            current_positions.append(pos)
            current_demand += demand
            last_pos = pos
    
    # Handle remaining nodes (tour might not end at depot)
    if current_nodes:
        current_distance += _euclidean_dist(last_pos, depot_coord)
        segments.append(Segment(
            nodes=current_nodes,
            total_demand=current_demand,
            total_distance=current_distance,
            start_pos=depot_coord,
            end_pos=depot_coord,
            node_positions=current_positions,
        ))
    
    return segments


def _euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class SegmentDistributor:
    """
    Distribute route segments to multiple vehicles.
    
    Supports multiple distribution strategies.
    """
    
    def __init__(
        self,
        strategy: str = "balanced",
        balance_weight: float = 0.5,  # Weight for distance vs demand balancing
    ):
        """
        Args:
            strategy: Distribution strategy
                - "sequential": Assign segments in order (first k to vehicle 0, etc.)
                - "round_robin": Alternate between vehicles
                - "balanced": Balance workload across vehicles
                - "distance_aware": Consider agent positions
                - "hungarian": Optimal assignment (requires scipy)
            balance_weight: Weight for balancing (0=demand only, 1=distance only)
        """
        self.strategy = strategy
        self.balance_weight = balance_weight
    
    def distribute(
        self,
        segments: List[Segment],
        num_vehicles: int,
        agent_infos: Optional[List[AgentInfo]] = None,
    ) -> List[List[Segment]]:
        """
        Distribute segments to vehicles.
        
        Args:
            segments: List of route segments
            num_vehicles: Number of vehicles
            agent_infos: Optional agent information for distance-aware assignment
            
        Returns:
            List of segment lists, one per vehicle
        """
        if not segments:
            return [[] for _ in range(num_vehicles)]
        
        if self.strategy == "sequential":
            return self._distribute_sequential(segments, num_vehicles)
        elif self.strategy == "round_robin":
            return self._distribute_round_robin(segments, num_vehicles)
        elif self.strategy == "balanced":
            return self._distribute_balanced(segments, num_vehicles)
        elif self.strategy == "distance_aware":
            return self._distribute_distance_aware(segments, num_vehicles, agent_infos)
        elif self.strategy == "hungarian":
            return self._distribute_hungarian(segments, num_vehicles, agent_infos)
        else:
            # Default to balanced
            return self._distribute_balanced(segments, num_vehicles)
    
    def _distribute_sequential(
        self,
        segments: List[Segment],
        num_vehicles: int,
    ) -> List[List[Segment]]:
        """Assign segments sequentially to vehicles."""
        result = [[] for _ in range(num_vehicles)]
        segments_per_vehicle = len(segments) // num_vehicles
        remainder = len(segments) % num_vehicles
        
        idx = 0
        for v in range(num_vehicles):
            count = segments_per_vehicle + (1 if v < remainder else 0)
            result[v] = segments[idx:idx + count]
            idx += count
        
        return result
    
    def _distribute_round_robin(
        self,
        segments: List[Segment],
        num_vehicles: int,
    ) -> List[List[Segment]]:
        """Distribute segments in round-robin fashion."""
        result = [[] for _ in range(num_vehicles)]
        for i, seg in enumerate(segments):
            result[i % num_vehicles].append(seg)
        return result
    
    def _distribute_balanced(
        self,
        segments: List[Segment],
        num_vehicles: int,
    ) -> List[List[Segment]]:
        """
        Balance workload across vehicles using greedy assignment.
        
        Assigns each segment to the vehicle with the smallest current workload.
        """
        result = [[] for _ in range(num_vehicles)]
        workloads = [0.0] * num_vehicles  # Combined workload metric
        
        # Sort segments by workload (descending) for better balance
        sorted_segments = sorted(
            segments,
            key=lambda s: self.balance_weight * s.total_distance + 
                         (1 - self.balance_weight) * s.total_demand,
            reverse=True,
        )
        
        for seg in sorted_segments:
            # Find vehicle with minimum workload
            min_idx = min(range(num_vehicles), key=lambda i: workloads[i])
            result[min_idx].append(seg)
            
            # Update workload
            seg_workload = (self.balance_weight * seg.total_distance + 
                          (1 - self.balance_weight) * seg.total_demand)
            workloads[min_idx] += seg_workload
        
        return result
    
    def _distribute_distance_aware(
        self,
        segments: List[Segment],
        num_vehicles: int,
        agent_infos: Optional[List[AgentInfo]] = None,
    ) -> List[List[Segment]]:
        """
        Distribute segments considering agent positions.
        
        Assigns segments to minimize total travel distance from agent positions.
        """
        if agent_infos is None or len(agent_infos) != num_vehicles:
            # Fall back to balanced if no agent info
            return self._distribute_balanced(segments, num_vehicles)
        
        result = [[] for _ in range(num_vehicles)]
        assigned = [False] * len(segments)
        agent_positions = [a.position for a in agent_infos]
        
        # Greedy assignment: for each segment, find best agent
        # Sort segments by some priority (e.g., total workload)
        segment_indices = sorted(
            range(len(segments)),
            key=lambda i: segments[i].total_distance + segments[i].total_demand,
            reverse=True,
        )
        
        # Track total distance traveled by each agent
        total_distances = [0.0] * num_vehicles
        
        for seg_idx in segment_indices:
            seg = segments[seg_idx]
            
            # Calculate cost for each agent
            costs = []
            for v in range(num_vehicles):
                # Cost = distance from current position to first node + segment distance
                # + balance penalty
                if seg.node_positions:
                    first_node_pos = seg.node_positions[0]
                else:
                    first_node_pos = seg.start_pos
                
                travel_cost = _euclidean_dist(agent_positions[v], first_node_pos)
                balance_penalty = total_distances[v] * 0.1  # Penalize overloaded agents
                costs.append(travel_cost + balance_penalty)
            
            # Assign to agent with minimum cost
            best_agent = min(range(num_vehicles), key=lambda i: costs[i])
            result[best_agent].append(seg)
            
            # Update agent position to end of segment
            if seg.node_positions:
                agent_positions[best_agent] = seg.node_positions[-1]
            total_distances[best_agent] += seg.total_distance
        
        return result
    
    def _distribute_hungarian(
        self,
        segments: List[Segment],
        num_vehicles: int,
        agent_infos: Optional[List[AgentInfo]] = None,
    ) -> List[List[Segment]]:
        """
        Use Hungarian algorithm for optimal segment assignment.
        
        This is useful when num_segments ≈ num_vehicles.
        For many segments, falls back to balanced distribution.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return self._distribute_balanced(segments, num_vehicles)
        
        if len(segments) <= num_vehicles:
            # Simple case: assign each segment to a vehicle
            result = [[] for _ in range(num_vehicles)]
            
            # Build cost matrix
            cost_matrix = np.zeros((len(segments), num_vehicles))
            
            for i, seg in enumerate(segments):
                for v in range(num_vehicles):
                    if agent_infos and v < len(agent_infos):
                        # Cost based on distance from agent to first node
                        if seg.node_positions:
                            first_pos = seg.node_positions[0]
                        else:
                            first_pos = seg.start_pos
                        cost_matrix[i, v] = _euclidean_dist(
                            agent_infos[v].position, first_pos
                        )
                    else:
                        cost_matrix[i, v] = seg.total_distance
            
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for seg_idx, vehicle_idx in zip(row_ind, col_ind):
                result[vehicle_idx].append(segments[seg_idx])
            
            return result
        else:
            # Many segments: use balanced with Hungarian for batches
            return self._distribute_balanced(segments, num_vehicles)


def convert_segments_to_routes(
    vehicle_segments: List[List[Segment]],
    include_depot: bool = True,
) -> List[List[int]]:
    """
    Convert segment assignments to route format.
    
    Args:
        vehicle_segments: Segments assigned to each vehicle
        include_depot: Whether to include depot visits (0) in routes
        
    Returns:
        Routes for each vehicle as list of node indices
    """
    routes = []
    for segments in vehicle_segments:
        route = []
        for seg in segments:
            if include_depot and route:
                route.append(0)  # Return to depot between segments
            route.extend(seg.nodes)
        if include_depot and route:
            route.append(0)  # End at depot
        routes.append(route)
    return routes


def get_segment_statistics(segments: List[Segment]) -> Dict[str, Any]:
    """Get statistics about segments."""
    if not segments:
        return {
            "count": 0,
            "total_nodes": 0,
            "total_demand": 0.0,
            "total_distance": 0.0,
            "avg_nodes_per_segment": 0.0,
            "avg_demand_per_segment": 0.0,
            "avg_distance_per_segment": 0.0,
        }
    
    total_nodes = sum(len(s.nodes) for s in segments)
    total_demand = sum(s.total_demand for s in segments)
    total_distance = sum(s.total_distance for s in segments)
    
    return {
        "count": len(segments),
        "total_nodes": total_nodes,
        "total_demand": total_demand,
        "total_distance": total_distance,
        "avg_nodes_per_segment": total_nodes / len(segments),
        "avg_demand_per_segment": total_demand / len(segments),
        "avg_distance_per_segment": total_distance / len(segments),
    }


def get_distribution_balance(
    vehicle_segments: List[List[Segment]],
) -> Dict[str, Any]:
    """Calculate balance metrics for segment distribution."""
    workloads = []
    for segments in vehicle_segments:
        distance = sum(s.total_distance for s in segments)
        demand = sum(s.total_demand for s in segments)
        nodes = sum(len(s.nodes) for s in segments)
        workloads.append({
            "distance": distance,
            "demand": demand,
            "nodes": nodes,
            "segments": len(segments),
        })
    
    if not workloads:
        return {"cv_distance": 0.0, "cv_demand": 0.0, "cv_nodes": 0.0}
    
    # Calculate coefficient of variation for each metric
    def cv(values):
        if not values or max(values) == 0:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return (variance ** 0.5) / mean
    
    return {
        "cv_distance": cv([w["distance"] for w in workloads]),
        "cv_demand": cv([w["demand"] for w in workloads]),
        "cv_nodes": cv([w["nodes"] for w in workloads]),
        "workloads": workloads,
    }
