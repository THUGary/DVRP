"""
Global Optimization Planner for DVRP.

Implements several optimization strategies:
1. Branch-and-Bound with early termination
2. Simulated Annealing for tour improvement
3. Or-opt / 2-opt local search
4. Cluster-first, route-second approach

Goal: Find near-optimal routes that minimize total travel distance
while respecting vehicle capacity constraints.
"""

from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Set
from collections import deque
import random
import math
import time
from .base import BasePlanner, AgentState, Target


class GlobalOptimizationPlanner(BasePlanner):
    """
    A global optimization planner that uses multiple strategies
    to find near-optimal vehicle routes.
    
    Modes:
    - "cluster_tsp": Cluster demands then solve TSP per cluster
    - "sa": Simulated Annealing for global optimization
    - "branch_bound": Branch and bound (exact for small instances)
    - "hybrid": Combination of clustering + local search
    """
    
    def __init__(
        self,
        full_capacity: int | None = None,
        *,
        mode: str = "hybrid",
        time_limit: float = 0.05,  # Max planning time in seconds
        sa_initial_temp: float = 100.0,
        sa_cooling_rate: float = 0.995,
        sa_iterations: int = 1000,
        local_search_iterations: int = 100,
    ) -> None:
        super().__init__()
        if full_capacity is None:
            raise RuntimeError("GlobalOptimizationPlanner requires full_capacity")
        self.full_capacity = int(full_capacity)
        self.mode = mode.lower()
        self.time_limit = time_limit
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations = sa_iterations
        self.local_search_iterations = local_search_iterations
        self._large_cost = 1e9
    
    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        """
        Main planning interface.
        """
        start_time = time.time()
        
        # Build unique demand list with coordinates and demands
        demands: List[Tuple[int, int, int]] = []  # (x, y, demand_amount)
        seen: Set[Tuple[int, int]] = set()
        for (x, y, t_arrival, c, t_due) in observations:
            if c <= 0:
                continue
            key = (x, y)
            if key not in seen:
                seen.add(key)
                demands.append((x, y, int(c)))
        
        num_agents = len(agent_states)
        
        # If no demands, all agents return to depot
        if not demands:
            return [deque([depot]) for _ in range(num_agents)]
        
        # Get agent positions and capacities
        agent_positions = [(a.x, a.y) for a in agent_states]
        agent_capacities = [a.s for a in agent_states]
        
        # Select optimization strategy
        if self.mode == "cluster_tsp":
            routes = self._cluster_tsp_optimize(
                demands, agent_positions, agent_capacities, depot, start_time
            )
        elif self.mode == "sa":
            routes = self._simulated_annealing_optimize(
                demands, agent_positions, agent_capacities, depot, start_time
            )
        elif self.mode == "branch_bound":
            routes = self._branch_bound_optimize(
                demands, agent_positions, agent_capacities, depot, start_time
            )
        else:  # hybrid
            routes = self._hybrid_optimize(
                demands, agent_positions, agent_capacities, depot, start_time
            )
        
        # Ensure all demands are covered (fallback to simple greedy if missing)
        routes = self._ensure_all_demands_covered(
            routes, demands, agent_positions, depot, num_agents
        )
        
        # Convert routes to target queues
        result: List[Deque[Target]] = []
        for i in range(num_agents):
            targets: Deque[Target] = deque()
            if i < len(routes):
                for (x, y, _) in routes[i]:
                    targets.append((x, y))
            if not targets:
                targets.append(depot)
            result.append(targets)
        
        return result
    
    def _ensure_all_demands_covered(
        self,
        routes: List[List[Tuple[int, int, int]]],
        demands: List[Tuple[int, int, int]],
        agent_positions: List[Tuple[int, int]],
        depot: Tuple[int, int],
        num_agents: int,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Ensure all demands are assigned to some route.
        If any are missing, append them to the route with least load
        or create new depot-return segments.
        """
        # Find which demands are already in routes
        assigned: Set[Tuple[int, int]] = set()
        for route in routes:
            for d in route:
                assigned.add((d[0], d[1]))
        
        # Find missing demands
        missing = [d for d in demands if (d[0], d[1]) not in assigned]
        
        if not missing:
            return routes
        
        # Ensure we have enough routes
        while len(routes) < num_agents:
            routes.append([])
        
        # Calculate current route loads
        route_loads = [self._route_demand(r) for r in routes]
        
        # Greedy assignment of missing demands
        for d in missing:
            dem = d[2]
            
            # Find route with least load that can fit this demand
            best_idx = None
            best_load = float('inf')
            
            for i in range(num_agents):
                # Check if adding this demand keeps route feasible
                # (allowing slight overflow is better than missing demands)
                if route_loads[i] < best_load:
                    best_load = route_loads[i]
                    best_idx = i
            
            if best_idx is not None:
                # Find best insertion position (nearest neighbor)
                route = routes[best_idx]
                if not route:
                    routes[best_idx] = [d]
                else:
                    # Find position that minimizes additional distance
                    best_pos = len(route)
                    best_increase = float('inf')
                    
                    d_coord = (d[0], d[1])
                    
                    # Try inserting at each position
                    for pos in range(len(route) + 1):
                        if pos == 0:
                            prev_coord = agent_positions[best_idx] if best_idx < len(agent_positions) else depot
                        else:
                            prev_coord = (route[pos-1][0], route[pos-1][1])
                        
                        if pos == len(route):
                            next_coord = depot
                        else:
                            next_coord = (route[pos][0], route[pos][1])
                        
                        # Cost of insertion
                        old_dist = self._distance(prev_coord, next_coord)
                        new_dist = self._distance(prev_coord, d_coord) + self._distance(d_coord, next_coord)
                        increase = new_dist - old_dist
                        
                        if increase < best_increase:
                            best_increase = increase
                            best_pos = pos
                    
                    route.insert(best_pos, d)
                
                route_loads[best_idx] += dem
        
        return routes
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Manhattan distance."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def _route_distance(
        self,
        route: List[Tuple[int, int, int]],
        start_pos: Tuple[int, int],
        depot: Tuple[int, int],
    ) -> int:
        """Calculate total distance of a route."""
        if not route:
            return 0
        total = self._distance(start_pos, (route[0][0], route[0][1]))
        for i in range(1, len(route)):
            total += self._distance(
                (route[i-1][0], route[i-1][1]),
                (route[i][0], route[i][1])
            )
        # Return to depot
        total += self._distance((route[-1][0], route[-1][1]), depot)
        return total
    
    def _route_demand(self, route: List[Tuple[int, int, int]]) -> int:
        """Total demand in a route."""
        return sum(d[2] for d in route)
    
    def _is_feasible(
        self,
        route: List[Tuple[int, int, int]],
        capacity: int,
    ) -> bool:
        """Check if route is feasible under capacity constraint."""
        return self._route_demand(route) <= capacity
    
    # ============== Cluster + TSP Approach ==============
    
    def _cluster_tsp_optimize(
        self,
        demands: List[Tuple[int, int, int]],
        agent_positions: List[Tuple[int, int]],
        agent_capacities: List[int],
        depot: Tuple[int, int],
        start_time: float,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        1. Cluster demands into groups that fit vehicle capacity
        2. Assign clusters to vehicles
        3. Solve TSP within each cluster
        """
        num_agents = len(agent_positions)
        
        # Calculate effective capacity for balanced distribution
        total_demand = sum(d[2] for d in demands)
        balanced_capacity = min(self.full_capacity, max(total_demand // num_agents + 10, 1))
        
        # Try sweep with balanced capacity and k-means
        sweep_clusters = self._sweep_clustering(demands, depot, balanced_capacity)
        kmeans_clusters = self._kmeans_clustering(demands, num_agents, self.full_capacity)
        
        # Evaluate both and pick the better one
        sweep_routes = self._assign_and_solve(
            sweep_clusters, agent_positions, agent_capacities, depot, num_agents
        )
        kmeans_routes = self._assign_and_solve(
            kmeans_clusters, agent_positions, agent_capacities, depot, num_agents
        )
        
        sweep_cost = self._total_solution_cost(sweep_routes, agent_positions, depot)
        kmeans_cost = self._total_solution_cost(kmeans_routes, agent_positions, depot)
        
        return sweep_routes if sweep_cost <= kmeans_cost else kmeans_routes
    
    def _assign_and_solve(
        self,
        clusters: List[List[Tuple[int, int, int]]],
        agent_positions: List[Tuple[int, int]],
        agent_capacities: List[int],
        depot: Tuple[int, int],
        num_agents: int,
    ) -> List[List[Tuple[int, int, int]]]:
        """Helper to assign clusters and solve TSP."""
        assignments = self._assign_clusters_to_agents(
            clusters, agent_positions, agent_capacities, depot
        )
        
        routes: List[List[Tuple[int, int, int]]] = [[] for _ in range(num_agents)]
        for agent_idx, cluster_indices in enumerate(assignments):
            combined_demands = []
            for cidx in cluster_indices:
                combined_demands.extend(clusters[cidx])
            
            if combined_demands:
                optimized = self._solve_tsp_nearest_neighbor(
                    combined_demands, agent_positions[agent_idx]
                )
                optimized = self._two_opt_improvement(
                    optimized, agent_positions[agent_idx], depot
                )
                routes[agent_idx] = optimized
        
        return routes
    
    def _kmeans_clustering(
        self,
        demands: List[Tuple[int, int, int]],
        k: int,
        capacity: int,
        max_iterations: int = 20,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        K-means style clustering that respects capacity constraints
        and tries to balance load across k clusters.
        """
        if not demands:
            return []
        
        n = len(demands)
        k = max(1, min(k, n))  # Can't have more clusters than demands
        
        # Calculate total demand to determine balanced cluster size
        total_demand = sum(d[2] for d in demands)
        # Use smaller effective capacity to force balanced distribution
        effective_capacity = min(capacity, max(total_demand // k + 5, 1))
        
        # Initialize centroids using k-means++ style selection
        centroids = []
        coords = [(d[0], d[1]) for d in demands]
        
        # First centroid is random
        first_idx = random.randrange(n)
        centroids.append(coords[first_idx])
        
        # Select remaining centroids proportional to squared distance
        for _ in range(1, k):
            dists = []
            for coord in coords:
                min_dist = min(self._distance(coord, c) for c in centroids)
                dists.append(min_dist ** 2)
            total = sum(dists)
            if total == 0:
                break
            r = random.uniform(0, total)
            cumsum = 0
            for i, d in enumerate(dists):
                cumsum += d
                if cumsum >= r:
                    centroids.append(coords[i])
                    break
        
        # Iterative assignment with balanced distribution
        for iteration in range(max_iterations):
            clusters: List[List[Tuple[int, int, int]]] = [[] for _ in range(len(centroids))]
            cluster_loads = [0] * len(centroids)
            
            # Sort demands by distance to nearest centroid (furthest first)
            def demand_priority(d):
                return min(self._distance((d[0], d[1]), c) for c in centroids)
            
            sorted_demands = sorted(demands, key=demand_priority, reverse=True)
            
            for d in sorted_demands:
                coord = (d[0], d[1])
                dem = d[2]
                
                # Find best cluster, preferring less loaded ones
                best_cluster = None
                best_score = float('inf')
                
                for cidx, centroid in enumerate(centroids):
                    # Use effective capacity for balanced distribution
                    if cluster_loads[cidx] + dem <= effective_capacity:
                        dist = self._distance(coord, centroid)
                        # Score: distance + penalty for load imbalance
                        load_penalty = cluster_loads[cidx] * 0.5
                        score = dist + load_penalty
                        if score < best_score:
                            best_score = score
                            best_cluster = cidx
                
                if best_cluster is not None:
                    clusters[best_cluster].append(d)
                    cluster_loads[best_cluster] += dem
                else:
                    # Overflow: assign to least loaded cluster
                    min_load_idx = min(range(len(centroids)), key=lambda i: cluster_loads[i])
                    clusters[min_load_idx].append(d)
                    cluster_loads[min_load_idx] += dem
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    cx = sum(d[0] for d in cluster) / len(cluster)
                    cy = sum(d[1] for d in cluster) / len(cluster)
                    new_centroids.append((cx, cy))
                else:
                    idx = len(new_centroids)
                    if idx < len(centroids):
                        new_centroids.append(centroids[idx])
            
            # Check convergence
            if len(new_centroids) == len(centroids):
                converged = all(
                    abs(new_centroids[i][0] - centroids[i][0]) < 1 and
                    abs(new_centroids[i][1] - centroids[i][1]) < 1
                    for i in range(len(centroids))
                )
                if converged:
                    break
            
            centroids = new_centroids
        
        # Remove empty clusters
        return [c for c in clusters if c]
    
    def _sweep_clustering(
        self,
        demands: List[Tuple[int, int, int]],
        depot: Tuple[int, int],
        capacity: int,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Sweep algorithm: sort demands by angle from depot,
        then group into capacity-feasible clusters.
        """
        if not demands:
            return []
        
        # Calculate angle from depot for each demand
        def angle_from_depot(d):
            dx = d[0] - depot[0]
            dy = d[1] - depot[1]
            return math.atan2(dy, dx)
        
        sorted_demands = sorted(demands, key=angle_from_depot)
        
        clusters = []
        current_cluster = []
        current_load = 0
        
        for d in sorted_demands:
            dem = d[2]
            if current_load + dem <= capacity:
                current_cluster.append(d)
                current_load += dem
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [d]
                current_load = dem
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _assign_clusters_to_agents(
        self,
        clusters: List[List[Tuple[int, int, int]]],
        agent_positions: List[Tuple[int, int]],
        agent_capacities: List[int],
        depot: Tuple[int, int],
    ) -> List[List[int]]:
        """
        Assign clusters to agents using greedy assignment
        based on distance from agent to cluster centroid.
        Ensures balanced distribution across agents.
        """
        num_agents = len(agent_positions)
        if num_agents == 0:
            return []
        
        assignments: List[List[int]] = [[] for _ in range(num_agents)]
        agent_loads = [0] * num_agents
        agent_node_counts = [0] * num_agents  # Track number of nodes per agent
        
        # Calculate cluster centroids and total demands
        cluster_info = []
        for cidx, cluster in enumerate(clusters):
            if not cluster:
                continue
            cx = sum(d[0] for d in cluster) / len(cluster)
            cy = sum(d[1] for d in cluster) / len(cluster)
            total_dem = sum(d[2] for d in cluster)
            num_nodes = len(cluster)
            cluster_info.append((cidx, cx, cy, total_dem, num_nodes))
        
        # Calculate target nodes per agent for balanced distribution
        total_nodes = sum(info[4] for info in cluster_info)
        target_nodes_per_agent = total_nodes / num_agents if num_agents > 0 else total_nodes
        
        # Sort clusters by size (larger first for better balancing)
        cluster_info.sort(key=lambda x: -x[4])
        
        for cidx, cx, cy, total_dem, num_nodes in cluster_info:
            # Find best agent, balancing load and distance
            best_agent = None
            best_score = float('inf')
            
            for aidx in range(num_agents):
                # Check capacity
                if agent_loads[aidx] + total_dem > self.full_capacity:
                    continue
                
                # Calculate distance cost
                if assignments[aidx]:
                    dist_cost = self._distance(depot, (int(cx), int(cy)))
                else:
                    dist_cost = self._distance(agent_positions[aidx], (int(cx), int(cy)))
                
                # Balance penalty: prefer agents with fewer nodes
                # Strong penalty for exceeding fair share
                balance_penalty = 0
                projected_nodes = agent_node_counts[aidx] + num_nodes
                if projected_nodes > target_nodes_per_agent * 1.5:
                    balance_penalty = (projected_nodes - target_nodes_per_agent) * 100
                else:
                    balance_penalty = agent_node_counts[aidx] * 2
                
                score = dist_cost + balance_penalty
                
                if score < best_score:
                    best_score = score
                    best_agent = aidx
            
            if best_agent is not None:
                assignments[best_agent].append(cidx)
                agent_loads[best_agent] += total_dem
                agent_node_counts[best_agent] += num_nodes
            else:
                # Fallback: assign to least loaded agent regardless of capacity
                min_nodes_agent = min(range(num_agents), key=lambda i: agent_node_counts[i])
                assignments[min_nodes_agent].append(cidx)
                agent_loads[min_nodes_agent] += total_dem
                agent_node_counts[min_nodes_agent] += num_nodes
        
        return assignments
    
    def _solve_tsp_nearest_neighbor(
        self,
        demands: List[Tuple[int, int, int]],
        start_pos: Tuple[int, int],
    ) -> List[Tuple[int, int, int]]:
        """Nearest neighbor heuristic for TSP."""
        if not demands:
            return []
        
        remaining = list(demands)
        route = []
        current = start_pos
        
        while remaining:
            best_idx = None
            best_dist = float('inf')
            for i, d in enumerate(remaining):
                dist = self._distance(current, (d[0], d[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx is not None:
                chosen = remaining.pop(best_idx)
                route.append(chosen)
                current = (chosen[0], chosen[1])
        
        return route
    
    def _two_opt_improvement(
        self,
        route: List[Tuple[int, int, int]],
        start_pos: Tuple[int, int],
        depot: Tuple[int, int],
        max_iterations: int = 50,
    ) -> List[Tuple[int, int, int]]:
        """Apply 2-opt local search to improve route."""
        if len(route) < 2:
            return route
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            best_distance = self._route_distance(route, start_pos, depot)
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Create new route by reversing segment [i+1, j]
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_distance = self._route_distance(new_route, start_pos, depot)
                    
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        return route
    
    # ============== Simulated Annealing ==============
    
    def _simulated_annealing_optimize(
        self,
        demands: List[Tuple[int, int, int]],
        agent_positions: List[Tuple[int, int]],
        agent_capacities: List[int],
        depot: Tuple[int, int],
        start_time: float,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Simulated annealing for global optimization of all routes.
        """
        num_agents = len(agent_positions)
        
        # Initialize with cluster-TSP solution
        current_routes = self._cluster_tsp_optimize(
            demands, agent_positions, agent_capacities, depot, start_time
        )
        current_cost = self._total_solution_cost(
            current_routes, agent_positions, depot
        )
        
        best_routes = [list(r) for r in current_routes]
        best_cost = current_cost
        
        temp = self.sa_initial_temp
        
        for iteration in range(self.sa_iterations):
            # Check time limit
            if time.time() - start_time > self.time_limit:
                break
            
            # Generate neighbor solution
            neighbor_routes = self._generate_neighbor(
                current_routes, num_agents, depot
            )
            
            # Check feasibility
            if not self._solution_feasible(neighbor_routes):
                continue
            
            neighbor_cost = self._total_solution_cost(
                neighbor_routes, agent_positions, depot
            )
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = neighbor_routes
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_routes = [list(r) for r in current_routes]
                    best_cost = current_cost
            
            # Cool down
            temp *= self.sa_cooling_rate
        
        return best_routes
    
    def _generate_neighbor(
        self,
        routes: List[List[Tuple[int, int, int]]],
        num_agents: int,
        depot: Tuple[int, int],
    ) -> List[List[Tuple[int, int, int]]]:
        """Generate a neighbor solution via random perturbation."""
        # Deep copy
        new_routes = [list(r) for r in routes]
        
        # Choose perturbation type
        move_type = random.choice(["relocate", "swap", "2opt", "cross"])
        
        if move_type == "relocate":
            # Move a demand from one route to another
            non_empty = [i for i, r in enumerate(new_routes) if r]
            if non_empty:
                src = random.choice(non_empty)
                if new_routes[src]:
                    idx = random.randrange(len(new_routes[src]))
                    demand = new_routes[src].pop(idx)
                    dst = random.randrange(num_agents)
                    if new_routes[dst]:
                        insert_pos = random.randrange(len(new_routes[dst]) + 1)
                    else:
                        insert_pos = 0
                    new_routes[dst].insert(insert_pos, demand)
        
        elif move_type == "swap":
            # Swap demands between two routes
            non_empty = [i for i, r in enumerate(new_routes) if r]
            if len(non_empty) >= 2:
                r1, r2 = random.sample(non_empty, 2)
                if new_routes[r1] and new_routes[r2]:
                    i1 = random.randrange(len(new_routes[r1]))
                    i2 = random.randrange(len(new_routes[r2]))
                    new_routes[r1][i1], new_routes[r2][i2] = \
                        new_routes[r2][i2], new_routes[r1][i1]
        
        elif move_type == "2opt":
            # 2-opt within a single route
            non_empty = [i for i, r in enumerate(new_routes) if len(r) >= 2]
            if non_empty:
                ridx = random.choice(non_empty)
                route = new_routes[ridx]
                if len(route) >= 2:
                    i = random.randrange(len(route) - 1)
                    j = random.randrange(i + 1, len(route))
                    route[i:j+1] = route[i:j+1][::-1]
        
        else:  # cross exchange
            # Exchange segments between two routes
            non_empty = [i for i, r in enumerate(new_routes) if r]
            if len(non_empty) >= 2:
                r1, r2 = random.sample(non_empty, 2)
                route1, route2 = new_routes[r1], new_routes[r2]
                if route1 and route2:
                    # Select segment from each route
                    i1 = random.randrange(len(route1))
                    j1 = random.randrange(i1, len(route1))
                    i2 = random.randrange(len(route2))
                    j2 = random.randrange(i2, len(route2))
                    
                    seg1 = route1[i1:j1+1]
                    seg2 = route2[i2:j2+1]
                    
                    new_routes[r1] = route1[:i1] + seg2 + route1[j1+1:]
                    new_routes[r2] = route2[:i2] + seg1 + route2[j2+1:]
        
        return new_routes
    
    def _solution_feasible(
        self,
        routes: List[List[Tuple[int, int, int]]],
    ) -> bool:
        """Check if all routes are capacity-feasible."""
        for route in routes:
            if self._route_demand(route) > self.full_capacity:
                return False
        return True
    
    def _total_solution_cost(
        self,
        routes: List[List[Tuple[int, int, int]]],
        agent_positions: List[Tuple[int, int]],
        depot: Tuple[int, int],
    ) -> int:
        """Total distance of all routes."""
        total = 0
        for i, route in enumerate(routes):
            if i < len(agent_positions):
                total += self._route_distance(route, agent_positions[i], depot)
            else:
                total += self._route_distance(route, depot, depot)
        return total
    
    # ============== Branch and Bound ==============
    
    def _branch_bound_optimize(
        self,
        demands: List[Tuple[int, int, int]],
        agent_positions: List[Tuple[int, int]],
        agent_capacities: List[int],
        depot: Tuple[int, int],
        start_time: float,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Branch and bound for small instances.
        Falls back to cluster-TSP if instance too large or time exceeded.
        """
        # For large instances, use heuristic
        if len(demands) > 15 or len(agent_positions) > 4:
            return self._cluster_tsp_optimize(
                demands, agent_positions, agent_capacities, depot, start_time
            )
        
        num_agents = len(agent_positions)
        
        # Get initial upper bound from heuristic
        heuristic_routes = self._cluster_tsp_optimize(
            demands, agent_positions, agent_capacities, depot, start_time
        )
        best_cost = self._total_solution_cost(heuristic_routes, agent_positions, depot)
        best_routes = heuristic_routes
        
        # Branch and bound search
        def bb_search(
            assigned: List[List[Tuple[int, int, int]]],
            remaining: List[Tuple[int, int, int]],
            loads: List[int],
            current_cost: int,
        ):
            nonlocal best_cost, best_routes
            
            if time.time() - start_time > self.time_limit:
                return
            
            if not remaining:
                # Complete solution
                total = self._total_solution_cost(assigned, agent_positions, depot)
                if total < best_cost:
                    best_cost = total
                    best_routes = [list(r) for r in assigned]
                return
            
            # Prune if current cost already exceeds best
            if current_cost >= best_cost:
                return
            
            demand = remaining[0]
            rest = remaining[1:]
            
            for agent_idx in range(num_agents):
                # Check capacity
                if loads[agent_idx] + demand[2] > self.full_capacity:
                    continue
                
                # Calculate incremental cost
                if assigned[agent_idx]:
                    last = assigned[agent_idx][-1]
                    inc_cost = self._distance((last[0], last[1]), (demand[0], demand[1]))
                else:
                    inc_cost = self._distance(agent_positions[agent_idx], (demand[0], demand[1]))
                
                # Branch
                new_assigned = [list(r) for r in assigned]
                new_assigned[agent_idx].append(demand)
                new_loads = list(loads)
                new_loads[agent_idx] += demand[2]
                
                bb_search(new_assigned, rest, new_loads, current_cost + inc_cost)
        
        # Start search
        initial_assigned = [[] for _ in range(num_agents)]
        initial_loads = [0] * num_agents
        bb_search(initial_assigned, demands, initial_loads, 0)
        
        return best_routes
    
    # ============== Hybrid Approach ==============
    
    def _hybrid_optimize(
        self,
        demands: List[Tuple[int, int, int]],
        agent_positions: List[Tuple[int, int]],
        agent_capacities: List[int],
        depot: Tuple[int, int],
        start_time: float,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Hybrid approach:
        1. Use cluster-TSP for initial solution
        2. Apply simulated annealing for global improvement
        3. Apply 2-opt for local improvement on each route
        """
        # Step 1: Get initial solution via clustering
        routes = self._cluster_tsp_optimize(
            demands, agent_positions, agent_capacities, depot, start_time
        )
        
        if time.time() - start_time > self.time_limit * 0.3:
            return routes
        
        # Step 2: Limited SA improvement
        remaining_time = self.time_limit - (time.time() - start_time)
        if remaining_time > 0.01:
            sa_iterations = min(self.sa_iterations, int(remaining_time * 5000))
            if sa_iterations > 50:
                routes = self._simulated_annealing_optimize(
                    demands, agent_positions, agent_capacities, depot, start_time
                )
        
        # Step 3: Final 2-opt polish on each route
        for i in range(len(routes)):
            if routes[i]:
                routes[i] = self._two_opt_improvement(
                    routes[i], agent_positions[i] if i < len(agent_positions) else depot, depot
                )
        
        return routes
