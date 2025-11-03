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
                 rng: random.Random, local_params: dict, env_params: dict, burst_params: dict) -> None:
        self.center_x = center[0]
        self.center_y = center[1]
        self.rng = rng
        self.local_params= local_params
        self.env_params = env_params
        self.burst_params = burst_params

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
        self.demands= self._generate_demands(self.local_params.get("distribution"),
                                             burst_mode=self.burst_params.get("burst_mode"))
        
    def sample(self, t: int) -> List[Demand]:
        """Sample demand points for current time step"""

        demand_t = []
        for demand in self.demands:
            if demand.t==t:
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
        count, time_series=self._sample_poisson_process(
            max_time=self.max_time, lambda_param=self.lambda_param)
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
        sample_count=int(count*2)
        if distribution == "uniform":
            samples = self._sample_uniform_2d(sample_count)
        elif distribution == "gaussian":
            samples= self._sample_gaussian_2d(sample_count)
        elif distribution == "cluster":
            samples = self._sample_cluster_2d(sample_count)
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
        sample_count=int(count*2)
        if distribution == "uniform":
            samples = self._sample_uniform_2d(sample_count,burst_mode=True)
        elif distribution == "gaussian":
            samples= self._sample_gaussian_2d(sample_count,burst_mode=True)
        elif distribution == "cluster":
            samples = self._sample_cluster_2d(sample_count,burst_mode=True)
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

    def remove_in_depot(self, pts:List[Tuple[int,int]]) -> bool:
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
        gx = np.random.randint(int(x_low), int(x_high) + 1, size=n_points)
        gy = np.random.randint(int(y_low), int(y_high) + 1, size=n_points)
        # print(f"Uniform samples x in [{x_low},{x_high}], y in [{y_low},{y_high}]")
        return np.column_stack((gx, gy))

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

        points= np.random.multivariate_normal(mean, cov , size=n_points)
        gx = np.floor(points[:, 0]).astype(int)
        gy = np.floor(points[:, 1]).astype(int)
        gx = np.clip(gx, 0, int(self.width) - 1)
        gy = np.clip(gy, 0, int(self.height) - 1)
        return np.column_stack((gx, gy))
    
    def _sample_cluster_2d(self, n_points: int, burst_mode:bool=False) -> np.ndarray:
        """sample points in 2D with exponential decay from center"""

        scale_factor= self.local_params.get("scale_factor")
        if scale_factor is None:
            print("No cluster distribution params!")
            return None
        
        if burst_mode:
            scale_factor=np.sqrt(n_points)/5.0
        
        W, H = self.width, self.height
        
        # create distance grid
        x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
        distances = np.sqrt((x_coords - self.center_x)**2 + (y_coords - self.center_y)**2)

        # compute exponential decay probabilities
        probabilities = np.exp(-distances / scale_factor).flatten()
        probabilities /= probabilities.sum()  # normalize

        # sample according to probabilities
        total_cells = W * H
        indices = np.random.choice(total_cells, size=n_points, replace=(n_points > total_cells), p=probabilities)
        
        # index to (x,y)
        x_selected = indices % W
        y_selected = indices // W
        
        return np.column_stack((x_selected, y_selected))



class RuleBasedGenerator(BaseDemandGenerator):
    """Generate demand points in rules."""

    def reset(self, seed: Optional[int] = None) -> None:
        seed = seed if seed is not None else self.params.get("rng_seed")
        self._rng = random.Random(seed)

        # Initialize concentrated generation areas
        self.neighborhoods = self._initialize_neighborhoods()
        
    def _initialize_neighborhoods(self) -> List[Neighborhood]:
        """Initialize concentrated generation areas"""
        num_centers = self.params.get("num_centers", 3)  # Number of center points
        
        neighborhoods = []
        for _ in range(num_centers):
           # Sample center coordinates
            center_x = self._rng.uniform(0, self.width)
            center_y = self._rng.uniform(0, self.height)
            local_max_c=self.params.get("max_c")
            lambda_param=self.total_demand/num_centers/self.params.get("max_time")/(1+local_max_c/2)
            
            distribution=self.params.get("distribution")
            size=self.params.get("neighborhood_size",3)
            size=max(3,size)
            
            if distribution=="uniform":
                distribution_params={
                    "distribution":"uniform",
                    "size":random.uniform(0.25*size,1.25*size),
                    }
            elif distribution=="gaussian":
                distribution_params={
                    "distribution":"gaussian",
                    "sigma1":random.uniform(0.5*size/3,1.5*size/3),
                    "sigma2":random.uniform(0.5*size/3,1.5*size/3),
                    "rho":0.0,
                    }
            elif distribution=="cluster":
                distribution_params={
                    "distribution":"cluster",
                    "scale_factor":size/5.0,
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
            )
            neighborhoods.append(neighborhood)
        
        return neighborhoods

    def sample(self, t: int) -> List[Demand]:
        """Sample all demand points at the current time step."""
        # print(f"Remaining total_demand: {self.total_demand}")

        if getattr(self, "total_demand", 0) <= 0:
            return []
        
        all_demands = []

        # Sample demand points from all concentrated generation areas
        # and resample those coinciding with depot (with attempt limit)
        max_tries = int(self.params.get("resample_depot_overlap_max_tries", 8))
        depot_xy = tuple(self.depot) if getattr(self, "depot", None) is not None else None
        for neighborhood in self.neighborhoods:
            demands = neighborhood.sample(t)
            if depot_xy is None or len(demands) == 0:
                all_demands.extend(demands)
                continue

            dx, dy = depot_xy
            for d in demands:
                if d.x == dx and d.y == dy:
                    # resample location up to max_tries
                    new_xy = (d.x, d.y)
                    ok = False
                    for _ in range(max_tries):
                        sx, sy = neighborhood.sample_one_xy()
                        if sx == dx and sy == dy:
                            continue
                        new_xy = (sx, sy)
                        ok = True
                        break
                    if ok:
                        all_demands.append(Demand(x=int(new_xy[0]), y=int(new_xy[1]), t=d.t, c=d.c, end_t=d.end_t))
                    else:
                        # give up: drop this demand (rare)
                        continue
                else:
                    all_demands.append(d)

        # Merge demands fallen into the same grid cell
        merged_demands = self._merge_demands_by_grid(all_demands)
        
        # After per-neighborhood resampling, merged_demands should no longer contain depot-overlapping points

        # strictly control total demand quantity
        total_c = sum(d.c for d in merged_demands)
        remove_ids=[]
    
        while total_c > self.total_demand:
            id=random.randint(0,len(merged_demands)-1)
            total_c -= merged_demands[id].c
            remove_ids.append(id)

        self.total_demand -= total_c

        merged_demands = [d for i, d in enumerate(merged_demands) if i not in remove_ids]

        return merged_demands
    
    def _merge_demands_by_grid(self, demands: List[Demand]) -> List[Demand]:
        """Merge demands fallen into the same grid cell."""
       
        merged_demands: Dict[Tuple[int, int, int], Demand] = {}
        max_c = self.params.get("max_c")
        
        for demand in demands:
            key = (demand.x, demand.y, demand.t)
            if key not in merged_demands:
                merged_demands[key] = demand
            else:
                existing_demand = merged_demands[key]
                new_c = min(existing_demand.c + demand.c, max_c)
                new_end_t = max(existing_demand.end_t, demand.end_t)
                
                # Create a new Demand object instead of modifying the old one
                merged_demands[key] = Demand(
                    x=demand.x,
                    y=demand.y,
                    t=demand.t,
                    c=new_c,
                    end_t=new_end_t
                )

        return list(merged_demands.values())