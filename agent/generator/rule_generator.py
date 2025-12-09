from __future__ import annotations
from os import times
import random
import math
from typing import List, Optional, Dict, Tuple
from .base import BaseDemandGenerator, Demand
import numpy as np


class Neighborhood:
    """Neighborhood for generating demands (2Dpositions, timestamps, quantities, lifetimes).\n
    **require params:**
    - center coordinates in `(center_x, center_y)`
    - rng: random.Random instance
    - local_params: dict, keys including: `distribution` (with its related parameters), \n
        `lambda_param`, `max_c`, `min_lifetime`, `max_lifetime`
    - env_params: dict, keys including: `width`, `height`, `depot`, `max_time`
    - burst_params: dict, keys including: `burst_mode` (bool), `burst_prob` (float, 0~1)
    """
    
    def __init__(self, center: tuple [float, float],  
                 rng: random.Random, local_params: dict, env_params: dict, burst_params: dict,
                 target_num_nodes: Optional[int] = None) -> None:
        self.center_x = center[0]
        self.center_y = center[1]
        self.rng = rng
        self.local_params= local_params
        self.env_params = env_params
        self.burst_params = burst_params
        # If provided, generate exactly this many demands (used when limiting by total node count)
        self.target_num_nodes = target_num_nodes

        # Poisson Process parameters
        self.lambda_param = local_params.get("lambda_param", 0.5)
        
		# Demand generation parameters
        self.max_c = int(local_params.get("max_c", 1))
        self.min_lifetime = int(local_params.get("min_lifetime", 5))
        self.max_lifetime = int(local_params.get("max_lifetime", 15))
        self.max_time=float(env_params.get("max_time",10))
        self.width=env_params.get("width",10)
        self.height=env_params.get("height",10)
        

        # Generate Basic demands and Burst demands IN ADVANCE
        # If `target_num_nodes` is provided, generate exactly that many demands evenly across time
        self.demands= self._generate_demands(self.local_params.get("distribution"),
                             burst_mode=self.burst_params.get("burst_mode"))
        # print(f"Generated total {len(self.demands)} demands in neighborhood centered at ({self.center_x}, {self.center_y}).")

    def sample(self, t: int) -> List[Demand]:
        """Sample demand points for current time step"""

        demand_t = []
        # print(f"Sampling demands at time {t}...demands left: {len(self.demands)}")
        for demand in self.demands:
            if demand.t==t:
                # print(f"Sampled demand at ({demand.x}, {demand.y}) with quantity {demand.c} and time {demand.t}")
                demand_t.append(demand)
                
        # remove sampled basic demands, which always come first in the list
        num_sampled=len(demand_t)
        self.demands=self.demands[num_sampled:]

        return demand_t

    def sample_one_xy(self) -> Tuple[int, int]:
        """Sample a single (x,y) according to this neighborhood's configured distribution.
        Falls back to uniform on the whole grid if distribution params are missing.
        """
        dist = self.local_params.get("distribution")
        xy = None
        try:
            if dist == "uniform":
                arr = self._sample_uniform_2d(1)
                xy = (int(arr[0, 0]), int(arr[0, 1])) if arr is not None and len(arr) > 0 else None
            elif dist == "gaussian":
                arr = self._sample_gaussian_2d(1)
                xy = (int(arr[0, 0]), int(arr[0, 1])) if arr is not None and len(arr) > 0 else None
            elif dist == "cluster":
                arr = self._sample_cluster_2d(1)
                xy = (int(arr[0, 0]), int(arr[0, 1])) if arr is not None and len(arr) > 0 else None
            elif dist == "explosion":##########
                arr = self._sample_explosion_2d(1)
                xy = (int(arr[0, 0]), int(arr[0, 1])) if arr is not None and len(arr) > 0 else None
            elif dist == "implosion":#########
                arr = self._sample_implosion_2d(1)
                xy = (int(arr[0, 0]), int(arr[0, 1])) if arr is not None and len(arr) > 0 else None
        except Exception:
            xy = None

        if xy is None:
            # fallback: uniform over the whole map
            x = int(np.random.randint(0, max(1, int(self.width))))
            y = int(np.random.randint(0, max(1, int(self.height))))
            xy = (x, y)
        return xy

    def _sample_poisson_process(self,max_time:int, lambda_param: float) -> Tuple[int, np.ndarray]:
        """Sample demand temporal points using Poisson process.\n
        returns events_count, time_series in chronical order. 
        """

        if lambda_param <= 0 or max_time <= 0:
            return 0, np.array([])

        events_count = np.random.poisson(lambda_param * max_time)

        # Handle the case where no events are generated
        if events_count == 0:
            return 0, np.array([])

        time_series = np.random.randint(0, max_time, size=events_count)
        time_series.sort()
        time_series = time_series - time_series[0]

        return events_count, time_series

    def _generate_demands(self, distribution: str, burst_mode: bool=False) -> List[Demand]:
        """Generate demands according to the specified distribution.\n
        returns basic_demands, burst_demands (empty if burst_mode is False)
        """
        # If a fixed target number of nodes is requested, use deterministic sampling
        if getattr(self, 'target_num_nodes', None) is not None:
            return self._generate_given_num_demands(distribution, self.target_num_nodes)

        count, time_series=self._sample_poisson_process(
            max_time=self.max_time, lambda_param=self.lambda_param)

        # If the Poisson process produced zero events, retry a few times to
        # avoid completely empty neighborhoods. If after retries there are
        # still zero events, create a single fallback timestamp so the
        # neighborhood is not empty.
        if count == 0:
            max_retries = int(self.local_params.get("resample_attempts", 3))
            for _ in range(max_retries - 1):
                count, time_series = self._sample_poisson_process(
                    max_time=self.max_time, lambda_param=self.lambda_param)
                if count > 0:
                    break
            if count == 0:
                # Fallback: create one event at a random time within [0, max_time-1]
                mt = max(1, int(max(0, math.floor(self.max_time))))
                # prefer self.rng for reproducibility when available
                try:
                    t0 = int(self.rng.randint(0, mt - 1))
                except Exception:
                    t0 = int(np.random.randint(0, mt))
                count = 1
                time_series = np.array([t0], dtype=int)

        # print(f"time_series: {time_series}")
        if burst_mode:
            burst_prob = self.burst_params.get("burst_prob")
            if burst_prob is not None:
                burst_num=np.round(burst_prob*count).astype(int)
                burst_ids=np.random.choice(np.arange(count),size=burst_num,replace=False)
                burst_timestamps = time_series[burst_ids]
                burst_demands = self._burst_demand(distribution=distribution, time_series=burst_timestamps)
                basic_timestamps = np.delete(time_series, burst_ids)
            else:
                burst_prob=0.0
                burst_demands=[]
                basic_timestamps = time_series
                print("No burst probability parameter! Set to 0.0 by default.")
        else:
            burst_prob = 0.0
            burst_demands = []
            basic_timestamps = time_series
        
        basic_demands=self._basic_demands(distribution=distribution, time_series=basic_timestamps)
        if burst_demands:
            merged_demands=self.merge_list_by_ids(burst_demands, basic_demands, A_pos=burst_ids)
            return merged_demands
        else:
            return basic_demands

    def _generate_given_num_demands(self, distribution: str, n: int) -> List[Demand]:
        """Generate exactly `n` demands by evenly sampling times in [0, max_time)
        and sampling positions from the configured distribution. Excludes depot.
        Demand quantities c are sampled uniformly from 1..max_c.
        """
        if n <= 0:
            return []

        # Sample integer times uniformly at random from [0, max_time] (inclusive),
        # then shift the entire sequence so the minimum is 0.
        mt = max(0, int(math.floor(self.max_time)))
        # Use randint with high=mt+1 to include mt
        time_series = np.random.randint(0, mt + 1, size=n).astype(int)
        time_series.sort()
        # Shift so the first timestamp is 0
        t_min = int(time_series.min()) if len(time_series) > 0 else 0
        time_series = (time_series - t_min).astype(int)

        # Sample positions according to the neighborhood distribution
        # Use the appropriate sampler requesting `n` points when possible
        samples = None
        try:
            dist = self.local_params.get("distribution")
            if dist == "uniform":
                samples = self._sample_uniform_2d(n)
            elif dist == "gaussian":
                samples = self._sample_gaussian_2d(n)
            elif dist == "cluster":
                samples = self._sample_cluster_2d(n)
            elif dist == "explosion":
                samples = self._sample_explosion_2d(n)
            elif dist == "implosion":
                samples = self._sample_implosion_2d(n)
        except Exception:
            samples = None

        # Fallback to uniform sampling if distribution failed
        if samples is None:
            samples = self._sample_uniform_2d(n)

        # Remove any points coinciding with depot
        samples = self.remove_in_depot(samples)

        # If after removing depot we have fewer than n samples, resample the missing ones uniformly
        if len(samples) < n:
            missing = n - len(samples)
            extra = self._sample_uniform_2d(missing)
            extra = self.remove_in_depot(extra)
            if len(extra) > 0:
                samples = np.vstack((samples, extra)) if len(samples) > 0 else extra

        # Ensure length is exactly n
        if len(samples) > n:
            samples = samples[:n]

        demands: List[Demand] = []
        for i in range(n):
            px, py = int(samples[i, 0]), int(samples[i, 1])
            c = int(self.rng.randint(1, self.max_c)) if hasattr(self.rng, 'randint') else int(np.random.randint(1, self.max_c + 1))
            lifetime = int(self.rng.randint(self.min_lifetime, self.max_lifetime)) if hasattr(self.rng, 'randint') else int(np.random.randint(self.min_lifetime, self.max_lifetime + 1))
            end_t = int(time_series[i]) + int(lifetime)
            demand = Demand(x=px, y=py, t=int(time_series[i]), c=c, end_t=end_t)
            demands.append(demand)

        return demands

    def merge_list_by_ids(self,A:List, B:List, A_pos:List[int]) -> List:
        if len(A) != len(A_pos):
            raise ValueError("Length of A and A_pos must be the same.")
        if max(A_pos) >= len(A)+len(B):
            raise ValueError("A_pos exceeds the total length of merged list.")
        total_len=len(A)+len(B)
        merged_list=np.empty(total_len,dtype=object)
        merged_list[A_pos]=A

        mask=np.ones(total_len,dtype=bool)
        mask[A_pos]=False
        merged_list[mask]=B
        return merged_list.tolist()

    def _basic_demands(self, distribution: str, time_series: list[int]) -> List[Demand]:
        """Generate basic demands"""

        count=len(time_series)
        
        if distribution == "uniform":
            samples = self._sample_uniform_2d(count)
        elif distribution == "gaussian":
            samples= self._sample_gaussian_2d(count)
        elif distribution == "cluster":
            samples = self._sample_cluster_2d(count)
        elif distribution == "explosion":
            samples = self._sample_explosion_2d(count)
        elif distribution == "implosion":
            samples = self._sample_implosion_2d(count)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # remove depot-overlapping points
        samples=self.remove_in_depot(samples)
        if len(samples)>count:
            samples=samples[:count]

        demands = []
        for i, (px, py) in enumerate(samples):
            c = np.random.randint(1, self.max_c + 1)
            lifetime = np.random.randint(self.min_lifetime, self.max_lifetime + 1)
            end_t = time_series[i] + lifetime
            demand = Demand(x=px, y=py, t=time_series[i], c=c, end_t=end_t)
            demands.append(demand)
        return demands
    
    def _burst_demand(self,distribution: str, time_series:List[int]) -> List[Demand]:
        """Generate burst demands"""

        count=len(time_series)
        
        if distribution == "uniform":
            samples = self._sample_uniform_2d(count,burst_mode=True)
        elif distribution == "gaussian":
            samples= self._sample_gaussian_2d(count,burst_mode=True)
        elif distribution == "cluster":
            samples = self._sample_cluster_2d(count,burst_mode=True)
        elif distribution == "explosion":
            samples = self._sample_explosion_2d(count,burst_mode=True)
        elif distribution == "implosion":
            samples = self._sample_implosion_2d(count,burst_mode=True)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        demands = []

        # remove depot-overlapping points
        samples=self.remove_in_depot(samples)
        if len(samples)>count:
            samples=samples[:count]

        for i, (px,py) in enumerate(samples):
            c=self.max_c
            end_t=time_series[i]+self.max_lifetime
            demand = Demand(x=px, y=py, t=time_series[i], c=c, end_t=end_t)
            demands.append(demand)
        return demands

    def remove_in_depot(self, pts:List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        depot = self.env_params.get("depot")
        if depot is None:
            print("No depot info!")
            return pts
        pts=np.array(pts)
        depot=np.array(depot).reshape(1,2)
        mask=(pts[:,0]==depot[:,0]) & (pts[:,1]==depot[:,1])
        return pts[~mask]

    def _sample_uniform_2d(self,n_points:int, burst_mode: bool=False) -> tuple[float, float]:
        """sample uniform 2D points around the center"""
        
        size=self.local_params.get("size")
        if size is None:
            print("No uniform distribution params!")
            return None
        if burst_mode:
            size=np.ceil(np.sqrt(n_points))

        x_low=max(0,math.floor(self.center_x-size))
        x_high=min(self.width-1,math.ceil(self.center_x+size))
        y_low=max(0,math.floor(self.center_y-size))
        y_high=min(self.height-1,math.ceil(self.center_y+size))
        # Resample until we have `n_points` that are not in depot
        max_attempts = int(self.local_params.get("resample_attempts", 10))
        samples = np.empty((0, 2), dtype=int)
        attempts = 0
        while samples.shape[0] < n_points and attempts < max_attempts:
            need = n_points - samples.shape[0]
            gx = np.random.randint(int(x_low), int(x_high) + 1, size=need)
            gy = np.random.randint(int(y_low), int(y_high) + 1, size=need)
            new = np.column_stack((gx, gy))
            new = self.remove_in_depot(new)
            if new is not None and len(new) > 0:
                samples = np.vstack((samples, new)) if samples.shape[0] > 0 else new
            attempts += 1

        # Fallback: if still not enough, fill missing entries with neighborhood center
        if samples.shape[0] < n_points:
            missing = n_points - samples.shape[0]
            cx = int(np.clip(int(math.floor(self.center_x)), 0, int(self.width) - 1))
            cy = int(np.clip(int(math.floor(self.center_y)), 0, int(self.height) - 1))
            center_pts = np.tile(np.array([cx, cy], dtype=int), (missing, 1))
            center_pts = self.remove_in_depot(center_pts)
            if center_pts is not None and len(center_pts) > 0:
                samples = np.vstack((samples, center_pts)) if samples.shape[0] > 0 else center_pts
            # If still short (should be rare now), leave as-is — do not fallback to whole-map sampling

        if samples.shape[0] > n_points:
            samples = samples[:n_points]
        return samples

    def _sample_gaussian_2d(self,n_points:int, burst_mode: bool=False) -> tuple[float, float]:
        """sample a 2D Gaussian point around the center"""

        sigma1=self.local_params.get("sigma1")
        sigma2=self.local_params.get("sigma2")
        rho=self.local_params.get("rho") # 0 by default
        if sigma1 is None or sigma2 is None or rho is None:
            print("No Gaussian distribution params!")
            return None
        if burst_mode:
            sigma1=np.ceil(np.sqrt(n_points)/3)
            sigma2=sigma1

        mean=np.array([self.center_x, self.center_y])
        cov=rho * sigma1 * sigma2
        cov=np.array([[sigma1**2, cov],
                      [cov, sigma2**2]])

        # Resample until we have `n_points` that are not in depot
        max_attempts = int(self.local_params.get("resample_attempts", 10))
        samples = np.empty((0, 2), dtype=int)
        attempts = 0
        while samples.shape[0] < n_points and attempts < max_attempts:
            need = n_points - samples.shape[0]
            points = np.random.multivariate_normal(mean, cov, size=need)
            gx = np.floor(points[:, 0]).astype(int)
            gy = np.floor(points[:, 1]).astype(int)
            gx = np.clip(gx, 0, int(self.width) - 1)
            gy = np.clip(gy, 0, int(self.height) - 1)
            new = np.column_stack((gx, gy))
            new = self.remove_in_depot(new)
            if new is not None and len(new) > 0:
                samples = np.vstack((samples, new)) if samples.shape[0] > 0 else new
            attempts += 1

        # Fallback: if still not enough, fill missing entries with neighborhood center
        if samples.shape[0] < n_points:
            missing = n_points - samples.shape[0]
            cx = int(np.clip(int(math.floor(self.center_x)), 0, int(self.width) - 1))
            cy = int(np.clip(int(math.floor(self.center_y)), 0, int(self.height) - 1))
            center_pts = np.tile(np.array([cx, cy], dtype=int), (missing, 1))
            center_pts = self.remove_in_depot(center_pts)
            if center_pts is not None and len(center_pts) > 0:
                samples = np.vstack((samples, center_pts)) if samples.shape[0] > 0 else center_pts
            # If still short (should be rare now), leave as-is — do not fallback to whole-map sampling

        if samples.shape[0] > n_points:
            samples = samples[:n_points]
        return samples
    
    def _sample_cluster_2d(self, n_points: int, burst_mode:bool=False) -> np.ndarray:
        """sample points in 2D with exponential decay from center"""

        scale_factor = self.local_params.get("scale_factor", 1.0)
        if burst_mode:
            scale_factor = max(np.sqrt(n_points) / 5.0, 1e-3)  # 避免 scale_factor 太小导致概率溢出
        
        W, H = self.width, self.height

        # create distance grid
        x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
        distances = np.sqrt((x_coords - self.center_x) ** 2 + (y_coords - self.center_y) ** 2)

        # compute exponential decay probabilities
        probabilities = np.exp(-distances / scale_factor).flatten()
        
        # 防止全零导致 NaN
        if np.all(probabilities == 0):
            probabilities = np.ones_like(probabilities)
        probabilities /= probabilities.sum()

        # sample according to probabilities
        total_cells = W * H
        # Resample until we have `n_points` that are not in depot
        max_attempts = int(self.local_params.get("resample_attempts", 10))
        samples = np.empty((0, 2), dtype=int)
        attempts = 0
        while samples.shape[0] < n_points and attempts < max_attempts:
            need = n_points - samples.shape[0]
            indices = np.random.choice(total_cells, size=need, replace=(need > total_cells), p=probabilities)
            x_selected = indices % W
            y_selected = indices // W
            new = np.column_stack((x_selected, y_selected))
            new = self.remove_in_depot(new)
            if new is not None and len(new) > 0:
                samples = np.vstack((samples, new)) if samples.shape[0] > 0 else new
            attempts += 1

        # Fallback: if still not enough, fill missing entries with neighborhood center
        if samples.shape[0] < n_points:
            missing = n_points - samples.shape[0]
            cx = int(np.clip(int(math.floor(self.center_x)), 0, int(self.width) - 1))
            cy = int(np.clip(int(math.floor(self.center_y)), 0, int(self.height) - 1))
            center_pts = np.tile(np.array([cx, cy], dtype=int), (missing, 1))
            center_pts = self.remove_in_depot(center_pts)
            if center_pts is not None and len(center_pts) > 0:
                samples = np.vstack((samples, center_pts)) if samples.shape[0] > 0 else center_pts
            # If still short (should be rare now), leave as-is — do not fallback to whole-map sampling

        if samples.shape[0] > n_points:
            samples = samples[:n_points]
        return samples

    def _sample_explosion_2d(self, n_points: int, burst_mode:bool=False) -> np.ndarray:  ##############
        """sample points in 2D with exponential decay from center"""

        scale_factor= self.local_params.get("scale_factor")
        if scale_factor is None:
            print("No cluster distribution params!")
            return None
        
        if burst_mode:
            scale_factor=np.sqrt(n_points)/5.0
        
        W, H = self.width, self.height
        
        # distance grid
        x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
        distances = np.sqrt((x_coords - self.center_x)**2 + (y_coords - self.center_y)**2)

        # Explosion: probability increases with distance
        probabilities = 1 - np.exp(-distances / scale_factor)
        probabilities = probabilities.flatten()
        probabilities /= probabilities.sum()

        total_cells = W * H
        # Resample until we have `n_points` that are not in depot
        max_attempts = int(self.local_params.get("resample_attempts", 10))
        samples = np.empty((0, 2), dtype=int)
        attempts = 0
        while samples.shape[0] < n_points and attempts < max_attempts:
            need = n_points - samples.shape[0]
            indices = np.random.choice(total_cells, size=need, replace=(need > total_cells), p=probabilities)
            x_selected = indices % W
            y_selected = indices // W
            new = np.column_stack((x_selected, y_selected))
            new = self.remove_in_depot(new)
            if new is not None and len(new) > 0:
                samples = np.vstack((samples, new)) if samples.shape[0] > 0 else new
            attempts += 1

        # Fallback: if still not enough, fill missing entries with neighborhood center
        if samples.shape[0] < n_points:
            missing = n_points - samples.shape[0]
            cx = int(np.clip(int(math.floor(self.center_x)), 0, int(self.width) - 1))
            cy = int(np.clip(int(math.floor(self.center_y)), 0, int(self.height) - 1))
            center_pts = np.tile(np.array([cx, cy], dtype=int), (missing, 1))
            center_pts = self.remove_in_depot(center_pts)
            if center_pts is not None and len(center_pts) > 0:
                samples = np.vstack((samples, center_pts)) if samples.shape[0] > 0 else center_pts
            # If still short (should be rare now), leave as-is — do not fallback to whole-map sampling

        if samples.shape[0] > n_points:
            samples = samples[:n_points]
        return samples

    def _sample_implosion_2d(self, n_points: int, burst_mode: bool = False) -> np.ndarray:
        """sample points in 2D with 'implosion' pattern (sharply decaying towards center)"""

        scale_factor = self.local_params.get("scale_factor", 1.0)
        if burst_mode:
            scale_factor = max(np.sqrt(n_points) / 5.0, 1e-3)  # 避免 scale_factor 太小
        
        W, H = self.width, self.height

        # distance grid
        x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
        distances = np.sqrt((x_coords - self.center_x) ** 2 + (y_coords - self.center_y) ** 2)

        # Implosion: sharply decaying exponential
        probabilities = np.exp(-(distances / scale_factor) ** 2).flatten()

        # 防止全零导致 NaN
        if np.all(probabilities == 0):
            probabilities = np.ones_like(probabilities)
        probabilities /= probabilities.sum()

        total_cells = W * H
        # Resample until we have `n_points` that are not in depot
        max_attempts = int(self.local_params.get("resample_attempts", 10))
        samples = np.empty((0, 2), dtype=int)
        attempts = 0
        while samples.shape[0] < n_points and attempts < max_attempts:
            need = n_points - samples.shape[0]
            indices = np.random.choice(total_cells, size=need, replace=(need > total_cells), p=probabilities)
            x_selected = indices % W
            y_selected = indices // W
            new = np.column_stack((x_selected, y_selected))
            new = self.remove_in_depot(new)
            if new is not None and len(new) > 0:
                samples = np.vstack((samples, new)) if samples.shape[0] > 0 else new
            attempts += 1

        # Fallback: if still not enough, fill missing entries with neighborhood center
        if samples.shape[0] < n_points:
            missing = n_points - samples.shape[0]
            cx = int(np.clip(int(math.floor(self.center_x)), 0, int(self.width) - 1))
            cy = int(np.clip(int(math.floor(self.center_y)), 0, int(self.height) - 1))
            center_pts = np.tile(np.array([cx, cy], dtype=int), (missing, 1))
            center_pts = self.remove_in_depot(center_pts)
            if center_pts is not None and len(center_pts) > 0:
                samples = np.vstack((samples, center_pts)) if samples.shape[0] > 0 else center_pts
            # If still short (should be rare now), leave as-is — do not fallback to whole-map sampling

        if samples.shape[0] > n_points:
            samples = samples[:n_points]
        return samples


class RuleBasedGenerator(BaseDemandGenerator):
    """Generate demand points in rules."""

    def __init__(self, width: int, height: int, **params) -> None:
        super().__init__(width, height, **params)
        # Initialize by calling reset
        self.reset(params.get("rng_seed"))

    def reset(self, seed: Optional[int] = None) -> None:
        seed = seed if seed is not None else self.params.get("rng_seed")
        super().reset(seed)
        self._rng = random.Random(seed)
        
        # IMPORTANT: Also seed numpy's global random state for reproducibility.
        # Neighborhoods use np.random for Poisson sampling and other operations.
        if seed is not None:
            np.random.seed(seed)

        # Reset mutable counters from params BEFORE initializing neighborhoods
        # Support two limiting modes:
        # 1. num_nodes: limit by number of demand nodes (preferred)
        # 2. total_demand: limit by sum of all demand capacities (legacy)
        
        # num_nodes takes priority if provided
        self.remaining_nodes = self.params.get("num_nodes", None)
        if self.remaining_nodes is not None:
            self.remaining_nodes = int(self.remaining_nodes)
            self.limit_mode = "num_nodes"
        else:
            # Fallback to total_demand for backward compatibility
            try:
                self.total_demand = int(self.params.get("total_demand", 1))
            except Exception:
                self.total_demand = int(getattr(self, "total_demand", 1))
            self.limit_mode = "total_demand"

        # Track occupied positions across all time steps (one demand per position)
        self._occupied_positions: set = set()

        # Initialize concentrated generation areas
        self.neighborhoods = self._initialize_neighborhoods()
        
    def _initialize_neighborhoods(self) -> List[Neighborhood]:
        """Initialize concentrated generation areas"""
        num_centers = self.params.get("num_centers", 3)  # Number of center points
        
        neighborhoods = []
        max_time = self.params.get("max_time", 100)  # Default max_time if not provided
        # When limiting by number of nodes, distribute remaining nodes evenly across centers
        assigned_counts = None
        if self.limit_mode == "num_nodes":
            total = int(self.remaining_nodes) if getattr(self, 'remaining_nodes', None) is not None else 0
            # Assign each node independently to a neighborhood with equal probability.
            # Use numpy multinomial for a concise allocation; numpy's RNG is seeded in reset().
            if num_centers > 0 and total > 0:
                probs = [1.0 / num_centers] * num_centers
                assigned_counts = np.random.multinomial(total, probs).tolist()
            else:
                assigned_counts = [0] * num_centers

        for i in range(num_centers):
           # Sample center coordinates
            center_x = self._rng.uniform(0, self.width)
            center_y = self._rng.uniform(0, self.height)
            # Ensure neighborhood center is not equal to the depot (avoid creating a center exactly at depot)
            depot_xy = tuple(self.depot) if getattr(self, "depot", None) is not None else None
            if depot_xy is not None:
                depot_x, depot_y = depot_xy
                max_center_attempts = int(self.params.get("center_resample_attempts", 10))
                attempts = 0
                while int(center_x) == int(depot_x) and int(center_y) == int(depot_y) and attempts < max_center_attempts:
                    center_x = self._rng.uniform(0, self.width)
                    center_y = self._rng.uniform(0, self.height)
                    attempts += 1
                # If still equal after retries, nudge slightly away from depot
                if int(center_x) == int(depot_x) and int(center_y) == int(depot_y):
                    center_x = min(self.width - 1e-3, center_x + 0.5)
                    center_y = min(self.height - 1e-3, center_y + 0.5)
            local_max_c=self.params.get("max_c", 50)  # Default max_c if not provided
            
            # Compute lambda_param based on limit mode
            if self.limit_mode == "num_nodes":
                # We'll request a fixed number of nodes for this neighborhood instead of Poisson
                lambda_param = None
                target_n = assigned_counts[i] if assigned_counts is not None else 0
            else:
                # Legacy: use total_demand
                lambda_param = self.total_demand / num_centers / max_time / (1 + local_max_c / 2)
                target_n = None
            
            distribution=self.params.get("distribution")
            size=self.params.get("neighborhood_size",3)
            size=max(3,size)
            
            if distribution=="uniform":
                distribution_params={
                    "distribution":"uniform",
                    "size":self._rng.uniform(0.25*size,1.25*size),
                    }
            elif distribution=="gaussian":
                distribution_params={
                    "distribution":"gaussian",
                    "sigma1":self._rng.uniform(0.5*size/3,1.5*size/3),
                    "sigma2":self._rng.uniform(0.5*size/3,1.5*size/3),
                    "rho":0.0,
                    }
            elif distribution=="cluster":
                distribution_params={
                    "distribution":"cluster",
                    "scale_factor":size/5.0,
                }
            elif distribution == "explosion":########
                distribution_params = {
                "distribution": "explosion",
                "scale_factor": size/5.0,
                }
            elif distribution == "implosion":########
                distribution_params = {
                "distribution": "implosion",
                "scale_factor": size/5.0,
                }
            else:
                # Default to uniform distribution
                distribution_params = {
                    "distribution": "uniform",
                    "size": self._rng.uniform(0.25*size, 1.25*size),
                }

            local_params={
                "lambda_param":lambda_param,
                "max_c":local_max_c,
                "min_lifetime":self.params.get("min_lifetime",10),
                "max_lifetime":self.params.get("max_lifetime",25),
                **distribution_params,
            }
            env_params={
                "width":self.width,
                "height":self.height,
                "depot":self.depot,
                "max_time":self.max_time,
            }
            burst_params={
                "burst_mode": True if self.params.get("burst_prob", 0.0) > 0.0 else False,
                "burst_prob": self.params.get("burst_prob", 0.0),
            }
            neighborhood = Neighborhood(
                (center_x, center_y),
                rng=self._rng,
                local_params=local_params,
                env_params=env_params,
                burst_params=burst_params,
                target_num_nodes=target_n,
            )
            neighborhoods.append(neighborhood)
        
        return neighborhoods

    def sample(self, t: int) -> List[Demand]:
        """Sample all demand points at the current time step.
        
        Supports two limiting modes:
        - num_nodes mode: limit by total number of demand nodes
        - total_demand mode: limit by sum of all demand capacities (legacy)
        """
        # Check if we've exhausted our quota
        if self.limit_mode == "num_nodes":
            if getattr(self, "remaining_nodes", 0) <= 0:
                return []
        else:
            if getattr(self, "total_demand", 0) <= 0:
                return []
        
        all_demands = []

        # Sample demand points from all concentrated generation areas
        # and resample those coinciding with depot or already-occupied positions
        max_tries = int(self.params.get("resample_depot_overlap_max_tries", 8))
        depot_xy = tuple(self.depot) if getattr(self, "depot", None) is not None else None
        occupied = getattr(self, "_occupied_positions", set())
        
        for neighborhood in self.neighborhoods:
            demands = neighborhood.sample(t)
            if len(demands) == 0:
                continue

            dx, dy = depot_xy if depot_xy else (-1, -1)
            for d in demands:
                pos = (d.x, d.y)
                # Check if position is depot or already occupied
                needs_resample = (pos == (dx, dy)) or (pos in occupied)
                
                if needs_resample:
                    # resample location up to max_tries
                    new_xy = pos
                    ok = False
                    for _ in range(max_tries):
                        sx, sy = neighborhood.sample_one_xy()
                        candidate = (sx, sy)
                        if candidate == (dx, dy) or candidate in occupied:
                            continue
                        new_xy = candidate
                        ok = True
                        break
                    if ok:
                        occupied.add(new_xy)
                        all_demands.append(Demand(x=int(new_xy[0]), y=int(new_xy[1]), t=d.t, c=d.c, end_t=d.end_t))
                    else:
                        # give up: drop this demand (rare)
                        # print(f"Dropped overlapping demand at ({d.x}, {d.y}) after {max_tries} resample attempts.")
                        continue
                else:
                    occupied.add(pos)
                    all_demands.append(d)

        # Update occupied positions
        self._occupied_positions = occupied

        # Merge demands fallen into the same grid cell (should be rare now)
        merged_demands = self._merge_demands_by_grid(all_demands)
        
        # Apply limits based on mode
        if self.limit_mode == "num_nodes":
            # Limit by number of nodes
            num_demands = len(merged_demands)
            if num_demands > self.remaining_nodes:
                # Randomly select which demands to keep
                keep_indices = self._rng.sample(range(num_demands), self.remaining_nodes)
                merged_demands = [merged_demands[i] for i in sorted(keep_indices)]
            self.remaining_nodes -= len(merged_demands)
        else:
            # Legacy: limit by total demand capacity
            total_c = sum(d.c for d in merged_demands)
            remove_ids = []
        
            while total_c > self.total_demand:
                id = self._rng.randint(0, len(merged_demands) - 1)
                total_c -= merged_demands[id].c
                remove_ids.append(id)

            self.total_demand -= total_c
            merged_demands = [d for i, d in enumerate(merged_demands) if i not in remove_ids]

        enriched_demands = [
            Demand(
                x=d.x,
                y=d.y,
                t=d.t,
                c=d.c,
                end_t=d.end_t,
                service_time=self.sample_service_time(capacity=d.c),
            )
            for d in merged_demands
        ]

        return enriched_demands
    
    def _merge_demands_by_grid(self, demands: List[Demand]) -> List[Demand]:
        """Merge demands at the same grid position (x, y) regardless of time.
        This ensures only one demand exists per position.
        """
       
        merged_demands: Dict[Tuple[int, int], Demand] = {}
        max_c = self.params.get("max_c")
        
        for demand in demands:
            key = (demand.x, demand.y)  # Merge by position only, not time
            if key not in merged_demands:
                merged_demands[key] = demand
            else:
                existing_demand = merged_demands[key]
                # Keep earliest arrival time, merge capacity, extend deadline
                new_t = min(existing_demand.t, demand.t)
                new_c = min(existing_demand.c + demand.c, max_c)
                new_end_t = max(existing_demand.end_t, demand.end_t)
                
                merged_demands[key] = Demand(
                    x=demand.x,
                    y=demand.y,
                    t=new_t,
                    c=new_c,
                    end_t=new_end_t
                )

        return list(merged_demands.values())
