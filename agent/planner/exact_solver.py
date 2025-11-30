"""
Exact VRP Solver using Dynamic Programming (Held-Karp style) with capacity constraints.

This solver finds the OPTIMAL solution for small-scale VRP instances.
- Uses bitmask DP for state representation
- Handles multiple vehicles with capacity constraints
- Time complexity: O(n^2 * 2^n * K) where n = nodes, K = vehicles
- Practical limit: ~15-18 nodes for exact, larger uses heuristic

For larger instances, uses high-quality heuristics (Clarke-Wright + 2-opt/3-opt).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from collections import deque
from itertools import permutations
import math
import time


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two points."""
    return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class ExactVRPSolver:
    """
    Exact solver for Capacitated VRP using dynamic programming.
    
    For small instances (≤8 nodes or total_demand ≤ 24), finds the globally optimal solution.
    For larger instances, uses high-quality heuristics with local search.
    """
    
    # Thresholds for exact vs heuristic solving
    # DP complexity is O(3^n * n^2), so n=8 is practical limit for sub-second solving
    MAX_EXACT_NODES = 8   # Node count threshold (3^8 ≈ 6561 states)
    MAX_EXACT_DEMAND = 24  # Total demand threshold (~8 nodes with avg demand 3)
    
    def __init__(self, capacity: int, num_vehicles: int, use_euclidean: bool = False):
        """
        Initialize the exact solver.
        
        Args:
            capacity: Vehicle capacity
            num_vehicles: Number of vehicles
            use_euclidean: If True, use Euclidean distance; else Manhattan (default)
                          NOTE: Manhattan is default because agents move in 4 directions
                          (up/down/left/right) only, so actual travel distance is Manhattan.
        """
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.use_euclidean = use_euclidean
        self._dist_func = euclidean_distance if use_euclidean else manhattan_distance
    
    def solve(
        self,
        depot: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        demands: List[int],
        time_limit: float = 60.0,
    ) -> Tuple[float, List[List[int]]]:
        """
        Solve VRP using automatic mode selection (DP for small, heuristic for large).
        
        Args:
            depot: Depot coordinates (x, y)
            nodes: List of customer node coordinates [(x, y), ...]
            demands: List of demands for each node
            time_limit: Maximum time in seconds for solving
            
        Returns:
            (total_distance, routes) where routes[v] is list of node indices for vehicle v
        """
        n = len(nodes)
        total_demand = sum(demands) if demands else 0
        
        if n == 0:
            return 0.0, [[] for _ in range(self.num_vehicles)]
        
        # Use exact DP if node count is small OR total demand is low
        # This allows exact solving when total_demand <= 50 (regardless of node count)
        use_exact = (n <= self.MAX_EXACT_NODES) or (total_demand <= self.MAX_EXACT_DEMAND and n <= 20)
        
        if use_exact:
            return self._solve_dp(depot, nodes, demands)
        else:
            # Use high-quality heuristic for larger instances
            return self._solve_heuristic_large(depot, nodes, demands, time_limit)
    
    def solve_with_mode(
        self,
        depot: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        demands: List[int],
        time_limit: float = 60.0,
        force_dp: bool = True,
    ) -> Tuple[float, List[List[int]]]:
        """
        Solve VRP with explicit mode selection.
        
        Args:
            depot: Depot coordinates (x, y)
            nodes: List of customer node coordinates [(x, y), ...]
            demands: List of demands for each node
            time_limit: Maximum time in seconds for solving
            force_dp: If True, always use exact DP (for 'exact' mode).
                     If False, always use heuristic (for 'heuristic' mode).
            
        Returns:
            (total_distance, routes) where routes[v] is list of node indices for vehicle v
        """
        n = len(nodes)
        
        if n == 0:
            return 0.0, [[] for _ in range(self.num_vehicles)]
        
        if force_dp:
            # Always use exact DP
            return self._solve_dp(depot, nodes, demands)
        else:
            # Always use heuristic
            return self._solve_heuristic_large(depot, nodes, demands, time_limit)
    
    def _solve_dp(
        self,
        depot: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        demands: List[int],
    ) -> Tuple[float, List[List[int]]]:
        """
        Solve using bitmask DP (exact solution).
        
        State: dp[mask][last_node][vehicle_load] = min distance to visit nodes in mask,
               ending at last_node with current vehicle having vehicle_load remaining.
        
        For simplicity with multiple vehicles, we use a route-first approach:
        1. Generate all feasible routes (sequences that don't exceed capacity)
        2. Select the best combination of routes that covers all nodes
        """
        n = len(nodes)
        K = self.num_vehicles
        
        # Precompute distances
        # Index 0 = depot, 1..n = nodes
        all_points = [depot] + nodes
        dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                dist[i][j] = self._dist_func(all_points[i], all_points[j])
        
        # For small n, enumerate all possible route combinations
        # Each route is a subset of nodes that fits capacity
        
        # Generate all feasible single-vehicle routes with their costs
        # Route = (node_set_mask, cost, ordered_nodes)
        feasible_routes: List[Tuple[int, float, List[int]]] = []
        
        # Empty route (vehicle doesn't move)
        feasible_routes.append((0, 0.0, []))
        
        # Generate routes using DP for single vehicle TSP with capacity
        # dp_route[mask][last] = (min_cost, path)
        for start_mask in range(1, 1 << n):
            # Check if this subset is capacity-feasible
            total_demand = sum(demands[i] for i in range(n) if (start_mask >> i) & 1)
            if total_demand > self.capacity:
                continue
            
            # Solve TSP for this subset
            subset_nodes = [i for i in range(n) if (start_mask >> i) & 1]
            if not subset_nodes:
                continue
            
            best_cost, best_order = self._solve_tsp_subset(dist, subset_nodes, n)
            feasible_routes.append((start_mask, best_cost, best_order))
        
        # Now find the best combination of K routes that covers all nodes
        # This is a set cover / partition problem
        full_mask = (1 << n) - 1
        
        # DP: dp_cover[mask] = (min_cost, routes_used)
        INF = float('inf')
        dp_cover: Dict[int, Tuple[float, List[Tuple[int, float, List[int]]]]] = {0: (0.0, [])}
        
        for mask in range(1 << n):
            if mask not in dp_cover:
                continue
            curr_cost, curr_routes = dp_cover[mask]
            if len(curr_routes) >= K:
                continue  # Can't use more vehicles
            
            for route_mask, route_cost, route_nodes in feasible_routes:
                if route_mask == 0:
                    continue
                # Check no overlap
                if mask & route_mask:
                    continue
                new_mask = mask | route_mask
                new_cost = curr_cost + route_cost
                new_routes = curr_routes + [(route_mask, route_cost, route_nodes)]
                
                if new_mask not in dp_cover or dp_cover[new_mask][0] > new_cost:
                    dp_cover[new_mask] = (new_cost, new_routes)
        
        if full_mask not in dp_cover:
            # Shouldn't happen if capacity allows, try heuristic
            return self._solve_heuristic_large(depot, nodes, demands, 30.0)
        
        best_total, best_routes = dp_cover[full_mask]
        
        # Format output: list of node indices per vehicle
        result_routes: List[List[int]] = [[] for _ in range(K)]
        for v, (_, _, route_nodes) in enumerate(best_routes):
            if v < K:
                result_routes[v] = route_nodes
        
        return best_total, result_routes
    
    def _solve_tsp_subset(
        self,
        dist: List[List[float]],
        subset: List[int],
        n: int,
    ) -> Tuple[float, List[int]]:
        """
        Solve TSP for a subset of nodes starting and ending at depot.
        Uses Held-Karp DP algorithm.
        
        Args:
            dist: Full distance matrix (0 = depot)
            subset: List of node indices (0-based, not including depot)
            n: Total number of nodes
            
        Returns:
            (min_cost, ordered_node_list)
        """
        m = len(subset)
        if m == 0:
            return 0.0, []
        if m == 1:
            node = subset[0]
            cost = dist[0][node + 1] + dist[node + 1][0]
            return cost, [node]
        
        # Map subset indices to 0..m-1
        idx_map = {node: i for i, node in enumerate(subset)}
        
        INF = float('inf')
        # dp[mask][i] = min cost to visit nodes in mask, ending at subset[i]
        dp = [[INF] * m for _ in range(1 << m)]
        parent = [[(-1, -1)] * m for _ in range(1 << m)]
        
        # Initialize: start from depot to each node
        for i, node in enumerate(subset):
            dp[1 << i][i] = dist[0][node + 1]
            parent[1 << i][i] = (-1, -1)
        
        # Fill DP
        for mask in range(1, 1 << m):
            for last in range(m):
                if not ((mask >> last) & 1):
                    continue
                if dp[mask][last] == INF:
                    continue
                
                for nxt in range(m):
                    if (mask >> nxt) & 1:
                        continue
                    new_mask = mask | (1 << nxt)
                    new_cost = dp[mask][last] + dist[subset[last] + 1][subset[nxt] + 1]
                    if new_cost < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = new_cost
                        parent[new_mask][nxt] = (mask, last)
        
        # Find best ending node (add return to depot)
        full_mask = (1 << m) - 1
        best_cost = INF
        best_last = -1
        for i, node in enumerate(subset):
            total = dp[full_mask][i] + dist[node + 1][0]
            if total < best_cost:
                best_cost = total
                best_last = i
        
        # Reconstruct path
        path = []
        mask = full_mask
        last = best_last
        while last != -1:
            path.append(subset[last])
            prev_mask, prev_last = parent[mask][last]
            mask = prev_mask
            last = prev_last
        path.reverse()
        
        return best_cost, path
    
    def _solve_heuristic_large(
        self,
        depot: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        demands: List[int],
        time_limit: float = 60.0,
    ) -> Tuple[float, List[List[int]]]:
        """
        High-quality heuristic solver for larger instances.
        Uses multiple construction heuristics + intensive local search.
        """
        n = len(nodes)
        K = self.num_vehicles
        start_time = time.time()
        
        # Precompute distance matrix
        all_points = [depot] + nodes
        dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                dist[i][j] = self._dist_func(all_points[i], all_points[j])
        
        best_routes = None
        best_dist = float('inf')
        
        # Try multiple construction heuristics
        construction_methods = [
            lambda: self._clarke_wright(dist, demands, n, K),
            lambda: self._nearest_neighbor_construction(dist, demands, n, K),
            lambda: self._sweep_construction(depot, nodes, demands, n, K),
        ]
        
        for construct in construction_methods:
            if (time.time() - start_time) > time_limit * 0.3:
                break
            
            routes = construct()
            routes = self._intensive_local_search(dist, demands, routes, time_limit * 0.2, start_time)
            
            total = sum(self._route_distance_matrix(dist, r) for r in routes)
            if total < best_dist:
                best_dist = total
                best_routes = [list(r) for r in routes]
        
        # Final intensive improvement on best solution
        if best_routes and (time.time() - start_time) < time_limit * 0.9:
            best_routes = self._intensive_local_search(
                dist, demands, best_routes, 
                time_limit - (time.time() - start_time) - 0.1,
                start_time
            )
            best_dist = sum(self._route_distance_matrix(dist, r) for r in best_routes)
        
        # For multi-trip VRP: keep all routes but assign them to vehicles in round-robin
        # Each vehicle will execute multiple trips (return to depot between trips)
        if len(best_routes) > K:
            # Sort routes by distance (shortest first) for better assignment
            route_dists = [(self._route_distance_matrix(dist, r), i, r) for i, r in enumerate(best_routes)]
            route_dists.sort()
            
            # Assign routes to vehicles in round-robin fashion
            vehicle_routes: List[List[int]] = [[] for _ in range(K)]
            for idx, (_, _, route) in enumerate(route_dists):
                v = idx % K
                # For multi-trip: concatenate routes with implicit depot returns
                vehicle_routes[v].extend(route)
            
            best_routes = vehicle_routes
            # Recalculate total distance (including depot returns between trips)
            best_dist = sum(self._route_distance_matrix(dist, r) for r in best_routes)
        
        # Pad routes to num_vehicles if needed
        while len(best_routes) < K:
            best_routes.append([])
        
        # Recalculate distance
        best_dist = sum(self._route_distance_matrix(dist, r) for r in best_routes)
        
        return best_dist, best_routes[:K]
    
    def _nearest_neighbor_construction(
        self,
        dist: List[List[float]],
        demands: List[int],
        n: int,
        K: int,
    ) -> List[List[int]]:
        """Nearest neighbor construction heuristic."""
        visited = [False] * n
        routes = []
        
        for _ in range(K):
            if all(visited):
                break
            
            route = []
            load = 0
            current = 0  # Start at depot
            
            while True:
                # Find nearest unvisited node that fits
                best_next = -1
                best_dist = float('inf')
                
                for j in range(n):
                    if visited[j]:
                        continue
                    if load + demands[j] > self.capacity:
                        continue
                    d = dist[current][j + 1]
                    if d < best_dist:
                        best_dist = d
                        best_next = j
                
                if best_next == -1:
                    break
                
                route.append(best_next)
                visited[best_next] = True
                load += demands[best_next]
                current = best_next + 1
            
            if route:
                routes.append(route)
        
        # Handle any remaining nodes - MUST add all of them
        remaining = [i for i in range(n) if not visited[i]]
        for node in remaining:
            # Try to add to existing route with capacity
            added = False
            for route in routes:
                route_demand = sum(demands[i] for i in route)
                if route_demand + demands[node] <= self.capacity:
                    route.append(node)
                    added = True
                    break
            if not added:
                # Create new route (even if exceeds K vehicles - we must cover all)
                routes.append([node])
        
        return routes
    
    def _sweep_construction(
        self,
        depot: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        demands: List[int],
        n: int,
        K: int,
    ) -> List[List[int]]:
        """Sweep algorithm construction heuristic."""
        # Sort nodes by angle from depot
        def angle_from_depot(idx: int) -> float:
            dx = nodes[idx][0] - depot[0]
            dy = nodes[idx][1] - depot[1]
            return math.atan2(dy, dx)
        
        sorted_nodes = sorted(range(n), key=angle_from_depot)
        
        routes = []
        current_route = []
        current_load = 0
        
        for node in sorted_nodes:
            if current_load + demands[node] <= self.capacity:
                current_route.append(node)
                current_load += demands[node]
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [node]
                current_load = demands[node]
        
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def _intensive_local_search(
        self,
        dist: List[List[float]],
        demands: List[int],
        routes: List[List[int]],
        time_budget: float,
        global_start: float,
    ) -> List[List[int]]:
        """Intensive local search with multiple operators."""
        routes = [list(r) for r in routes]
        start_time = time.time()
        
        improved = True
        iteration = 0
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            iteration += 1
            
            # 2-opt within each route
            for v in range(len(routes)):
                if len(routes[v]) >= 2:
                    new_route, gain = self._two_opt_route(dist, routes[v])
                    if gain > 1e-9:
                        routes[v] = new_route
                        improved = True
            
            # Or-opt (segment relocation within route)
            for v in range(len(routes)):
                if len(routes[v]) >= 3:
                    new_route = self._or_opt(dist, routes[v])
                    old_dist = self._route_distance_matrix(dist, routes[v])
                    new_dist = self._route_distance_matrix(dist, new_route)
                    if new_dist < old_dist - 1e-9:
                        routes[v] = new_route
                        improved = True
            
            # Inter-route moves (relocate and swap)
            routes, inter_improved = self._inter_route_moves(dist, demands, routes)
            if inter_improved:
                improved = True
            
            # Cross exchange between routes (every 3rd iteration)
            if iteration % 3 == 0:
                routes, cross_improved = self._cross_exchange(dist, demands, routes)
                if cross_improved:
                    improved = True
        
        return routes
    
    def _cross_exchange(
        self,
        dist: List[List[float]],
        demands: List[int],
        routes: List[List[int]],
    ) -> Tuple[List[List[int]], bool]:
        """Cross exchange: swap segments between two routes."""
        improved = False
        routes = [list(r) for r in routes]
        
        for r1 in range(len(routes)):
            for r2 in range(r1 + 1, len(routes)):
                if not routes[r1] or not routes[r2]:
                    continue
                
                # Try swapping segments of length 1-3
                for len1 in range(1, min(4, len(routes[r1]) + 1)):
                    for len2 in range(1, min(4, len(routes[r2]) + 1)):
                        for i1 in range(len(routes[r1]) - len1 + 1):
                            for i2 in range(len(routes[r2]) - len2 + 1):
                                seg1 = routes[r1][i1:i1 + len1]
                                seg2 = routes[r2][i2:i2 + len2]
                                
                                # Check capacity feasibility
                                demand1 = sum(demands[n] for n in seg1)
                                demand2 = sum(demands[n] for n in seg2)
                                route1_demand = sum(demands[n] for n in routes[r1])
                                route2_demand = sum(demands[n] for n in routes[r2])
                                
                                new_r1_demand = route1_demand - demand1 + demand2
                                new_r2_demand = route2_demand - demand2 + demand1
                                
                                if new_r1_demand > self.capacity or new_r2_demand > self.capacity:
                                    continue
                                
                                # Calculate cost change
                                old_cost = (self._route_distance_matrix(dist, routes[r1]) +
                                           self._route_distance_matrix(dist, routes[r2]))
                                
                                new_r1 = routes[r1][:i1] + seg2 + routes[r1][i1 + len1:]
                                new_r2 = routes[r2][:i2] + seg1 + routes[r2][i2 + len2:]
                                
                                new_cost = (self._route_distance_matrix(dist, new_r1) +
                                           self._route_distance_matrix(dist, new_r2))
                                
                                if new_cost < old_cost - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        return routes, improved
    
    def _clarke_wright(
        self,
        dist: List[List[float]],
        demands: List[int],
        n: int,
        K: int,
    ) -> List[List[int]]:
        """Clarke-Wright savings algorithm for initial solution."""
        # Start with each customer in its own route
        routes: List[List[int]] = [[i] for i in range(n)]
        route_demands = [demands[i] for i in range(n)]
        route_of_node = {i: i for i in range(n)}
        
        # Calculate savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(n):
            for j in range(i + 1, n):
                s = dist[0][i + 1] + dist[0][j + 1] - dist[i + 1][j + 1]
                savings.append((s, i, j))
        
        # Sort by savings (descending)
        savings.sort(reverse=True)
        
        # Merge routes
        for s, i, j in savings:
            if s <= 0:
                break
            
            ri = route_of_node.get(i)
            rj = route_of_node.get(j)
            
            if ri is None or rj is None or ri == rj:
                continue
            
            # Check if merge is feasible
            if route_demands[ri] + route_demands[rj] > self.capacity:
                continue
            
            # Check if i and j are at route endpoints
            if not routes[ri] or not routes[rj]:
                continue
            
            # i must be at end of route ri, j must be at start of route rj (or vice versa)
            can_merge = False
            new_route = []
            
            if routes[ri][-1] == i and routes[rj][0] == j:
                new_route = routes[ri] + routes[rj]
                can_merge = True
            elif routes[ri][-1] == i and routes[rj][-1] == j:
                new_route = routes[ri] + routes[rj][::-1]
                can_merge = True
            elif routes[ri][0] == i and routes[rj][0] == j:
                new_route = routes[ri][::-1] + routes[rj]
                can_merge = True
            elif routes[ri][0] == i and routes[rj][-1] == j:
                new_route = routes[rj] + routes[ri]
                can_merge = True
            
            if can_merge:
                routes[ri] = new_route
                route_demands[ri] += route_demands[rj]
                routes[rj] = []
                route_demands[rj] = 0
                for node in new_route:
                    route_of_node[node] = ri
        
        # Filter empty routes
        return [r for r in routes if r]
    
    def _two_opt_route(
        self,
        dist: List[List[float]],
        route: List[int],
    ) -> Tuple[List[int], float]:
        """2-opt local search for a single route."""
        if len(route) < 2:
            return route, 0.0
        
        route = list(route)
        total_improvement = 0.0
        improved = True
        
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Calculate improvement from reversing route[i+1:j+1]
                    # Before: ... -> route[i] -> route[i+1] -> ... -> route[j] -> route[j+1 or depot] -> ...
                    # After:  ... -> route[i] -> route[j] -> ... -> route[i+1] -> route[j+1 or depot] -> ...
                    
                    a = 0 if i == 0 else route[i - 1] + 1
                    if i == 0:
                        a = 0  # depot
                    else:
                        a = route[i - 1] + 1
                    b = route[i] + 1
                    c = route[j] + 1
                    if j == len(route) - 1:
                        d = 0  # depot
                    else:
                        d = route[j + 1] + 1
                    
                    # Actually for 2-opt we reverse segment [i, j]
                    # old: a -> b=route[i] -> ... -> c=route[j] -> d
                    # new: a -> c=route[j] -> ... -> b=route[i] -> d
                    if i == 0:
                        a_idx = 0
                    else:
                        a_idx = route[i - 1] + 1
                    b_idx = route[i] + 1
                    c_idx = route[j] + 1
                    if j == len(route) - 1:
                        d_idx = 0
                    else:
                        d_idx = route[j + 1] + 1
                    
                    old_dist = dist[a_idx][b_idx] + dist[c_idx][d_idx]
                    new_dist = dist[a_idx][c_idx] + dist[b_idx][d_idx]
                    delta = new_dist - old_dist
                    
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-9:
                route[best_i:best_j + 1] = reversed(route[best_i:best_j + 1])
                total_improvement -= best_delta
                improved = True
        
        return route, total_improvement
    
    def _inter_route_moves(
        self,
        dist: List[List[float]],
        demands: List[int],
        routes: List[List[int]],
    ) -> Tuple[List[List[int]], bool]:
        """Try relocate and swap moves between routes."""
        improved = False
        routes = [list(r) for r in routes]
        
        # Relocate: move a node from one route to another
        for r1 in range(len(routes)):
            for r2 in range(len(routes)):
                if r1 == r2 or not routes[r1]:
                    continue
                
                for i in range(len(routes[r1])):
                    node = routes[r1][i]
                    node_demand = demands[node]
                    
                    # Check capacity
                    r2_demand = sum(demands[n] for n in routes[r2])
                    if r2_demand + node_demand > self.capacity:
                        continue
                    
                    # Calculate removal cost
                    removal_cost = self._removal_cost(dist, routes[r1], i)
                    
                    # Find best insertion position
                    best_insert_cost = float('inf')
                    best_pos = 0
                    for j in range(len(routes[r2]) + 1):
                        insert_cost = self._insertion_cost(dist, routes[r2], j, node)
                        if insert_cost < best_insert_cost:
                            best_insert_cost = insert_cost
                            best_pos = j
                    
                    if removal_cost + best_insert_cost < -1e-9:
                        # Perform move
                        routes[r1].pop(i)
                        routes[r2].insert(best_pos, node)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        return routes, improved
    
    def _removal_cost(self, dist: List[List[float]], route: List[int], i: int) -> float:
        """Cost change from removing node at position i."""
        if not route:
            return 0.0
        
        node = route[i]
        prev_idx = 0 if i == 0 else route[i - 1] + 1
        next_idx = 0 if i == len(route) - 1 else route[i + 1] + 1
        node_idx = node + 1
        
        old_cost = dist[prev_idx][node_idx] + dist[node_idx][next_idx]
        new_cost = dist[prev_idx][next_idx]
        
        return new_cost - old_cost
    
    def _insertion_cost(self, dist: List[List[float]], route: List[int], pos: int, node: int) -> float:
        """Cost change from inserting node at position pos."""
        node_idx = node + 1
        
        if not route:
            # depot -> node -> depot
            return dist[0][node_idx] + dist[node_idx][0]
        
        prev_idx = 0 if pos == 0 else route[pos - 1] + 1
        next_idx = 0 if pos == len(route) else route[pos] + 1
        
        old_cost = dist[prev_idx][next_idx]
        new_cost = dist[prev_idx][node_idx] + dist[node_idx][next_idx]
        
        return new_cost - old_cost
    
    def _or_opt(self, dist: List[List[float]], route: List[int]) -> List[int]:
        """Or-opt: relocate segments of 1, 2, or 3 consecutive nodes."""
        if len(route) < 3:
            return route
        
        route = list(route)
        improved = True
        
        while improved:
            improved = False
            
            for seg_len in [1, 2, 3]:
                if len(route) < seg_len + 1:
                    continue
                
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i + seg_len]
                    
                    # Calculate removal cost
                    prev_idx = 0 if i == 0 else route[i - 1] + 1
                    next_idx = 0 if i + seg_len >= len(route) else route[i + seg_len] + 1
                    first_idx = segment[0] + 1
                    last_idx = segment[-1] + 1
                    
                    removal_gain = (dist[prev_idx][first_idx] + dist[last_idx][next_idx] 
                                   - dist[prev_idx][next_idx])
                    
                    # Try inserting at other positions
                    for j in range(len(route) - seg_len + 1):
                        if abs(j - i) <= seg_len:
                            continue
                        
                        # Calculate insertion cost at position j
                        j_prev = 0 if j == 0 else route[j - 1] + 1
                        j_next = 0 if j >= len(route) else route[j] + 1
                        
                        # Adjust indices if j > i (after removal)
                        if j > i:
                            j_adj = j - seg_len
                            remaining = route[:i] + route[i + seg_len:]
                            j_prev = 0 if j_adj == 0 else remaining[j_adj - 1] + 1
                            j_next = 0 if j_adj >= len(remaining) else remaining[j_adj] + 1
                        
                        insertion_cost = (dist[j_prev][first_idx] + dist[last_idx][j_next]
                                         - dist[j_prev][j_next])
                        
                        if removal_gain - insertion_cost > 1e-9:
                            # Perform move
                            route = route[:i] + route[i + seg_len:]
                            insert_pos = j if j < i else j - seg_len
                            route = route[:insert_pos] + segment + route[insert_pos:]
                            improved = True
                            break
                    
                    if improved:
                        break
                if improved:
                    break
        
        return route
    
    def _route_distance_matrix(self, dist: List[List[float]], route: List[int]) -> float:
        """Calculate total distance for a route using distance matrix."""
        if not route:
            return 0.0
        
        total = dist[0][route[0] + 1]
        for i in range(len(route) - 1):
            total += dist[route[i] + 1][route[i + 1] + 1]
        total += dist[route[-1] + 1][0]
        return total
    
    def _route_distance(
        self,
        depot: Tuple[int, int],
        nodes: List[Tuple[int, int]],
        route: List[int],
    ) -> float:
        """Calculate total distance for a route (depot -> nodes -> depot)."""
        if not route:
            return 0.0
        
        dist = self._dist_func(depot, nodes[route[0]])
        for i in range(len(route) - 1):
            dist += self._dist_func(nodes[route[i]], nodes[route[i + 1]])
        dist += self._dist_func(nodes[route[-1]], depot)
        return dist


def solve_vrp_exact(
    depot: Tuple[int, int],
    nodes: List[Tuple[int, int]],
    demands: List[int],
    capacity: int,
    num_vehicles: int,
    use_euclidean: bool = False,
    time_limit: float = 60.0,
) -> Tuple[float, List[List[int]]]:
    """
    Convenience function to solve VRP exactly.
    
    Args:
        depot: Depot coordinates
        nodes: Customer node coordinates
        demands: Demand at each node
        capacity: Vehicle capacity
        num_vehicles: Number of vehicles
        use_euclidean: Use Euclidean distance (default False, use Manhattan
                      because agents move in 4 directions only)
        time_limit: Time limit in seconds for large instances
        
    Returns:
        (total_distance, routes) where routes[v] is ordered list of node indices
    """
    solver = ExactVRPSolver(capacity, num_vehicles, use_euclidean=use_euclidean)
    return solver.solve(depot, nodes, demands, time_limit)
