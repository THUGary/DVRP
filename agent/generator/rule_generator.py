from __future__ import annotations
from os import times
import random
import math
from typing import List, Optional, Dict, Tuple
from .base import BaseDemandGenerator, Demand
import numpy as np


class Neighborhood:
    """Neighborhood for generating demand points"""
    
    def __init__(self, center_x: float, center_y: float, width: int, height: int, 
                 rng: random.Random, params: dict) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.rng = rng
        self.params = params

        # Poisson Process parameters
        self.lambda_param = params.get("lambda_param", 0.5)
        
		# Demand generation parameters
        self.max_c = int(params.get("max_c", 1))
        self.min_lifetime = int(params.get("min_lifetime", 5))
        self.max_lifetime = int(params.get("max_lifetime", 15))
        self.max_time=float(params.get("max_time",10))
        
        # Generate demand point numbers for all time steps in advance
        self.events_count, self.time_series = self._sample_poisson_process(max_time=self.max_time, delta_t=1.0)
        # print(f"number of events sampled: {self.events_count}")
        self.demands=self._generate_demands(distribution=self.params.get("distribution"), count=self.events_count)
        
    def sample(self, t: int) -> List[Demand]:
        """Sample demand points for current time step"""

        demand_t = []
        for i, demand in enumerate(self.demands):
            if self.time_series[i] == t:
                demand_t.append(demand)
        return demand_t

    def _sample_poisson_process(self,max_time:float,delta_t:float=1.0) -> Tuple[int, np.ndarray]:
        """Sample demand temporal points using Poisson process"""
        
        if self.lambda_param <= 0 or max_time <= 0 or delta_t <= 0:
            return 0, np.array([])

        events_count=np.random.poisson(self.lambda_param * max_time)
        
        # Handle the case where no events are generated
        if events_count == 0:
            return 0, np.array([])

        time_series=np.random.randint(0, max_time, size=events_count)
        time_series.sort()
        time_series=time_series-time_series[0]

        return events_count, time_series

    def _generate_demands(self, distribution: str, count: int) -> List[Demand]:
        """Generate demand points at time t."""

        if distribution == "uniform":
            samples = self._sample_uniform_2d(count)
        elif distribution == "gaussian":
            samples= self._sample_gaussian_2d(count)
        elif distribution == "cluster":
            samples = self._sample_cluster_2d(count)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        demands = []
        for i, (px, py) in enumerate(samples):
            c = np.random.randint(1, self.max_c + 1)
            lifetime = np.random.randint(self.min_lifetime, self.max_lifetime + 1)
            end_t = self.time_series[i] + lifetime
            demand = Demand(x=px, y=py, t=self.time_series[i], c=c, end_t=end_t)
            demands.append(demand)
        return demands
    
    def _sample_uniform_2d(self,n_points:int) -> tuple[float, float]:
        """sample uniform 2D points around the center"""
        
        size=self.params.get("size")
        if size is None:
            print("No uniform distribution params!")
            return None

        x_low=max(0,math.floor(self.center_x-size))
        x_high=min(self.width-1,math.ceil(self.center_x+size))
        y_low=max(0,math.floor(self.center_y-size))
        y_high=min(self.height-1,math.ceil(self.center_y+size))
        gx = np.random.randint(int(x_low), int(x_high) + 1, size=n_points)
        gy = np.random.randint(int(y_low), int(y_high) + 1, size=n_points)
        # print(f"Uniform samples x in [{x_low},{x_high}], y in [{y_low},{y_high}]")
        return np.column_stack((gx, gy))

    def _sample_gaussian_2d(self,n_points:int) -> tuple[float, float]:
        """sample a 2D Gaussian point around the center"""

        sigma1=self.params.get("sigma1")
        sigma2=self.params.get("sigma2")
        rho=self.params.get("rho")
        if sigma1 is None or sigma2 is None or rho is None:
            print("No Gaussian distribution params!")
            return None

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
    
    def _sample_cluster_2d(self, n_points: int) -> np.ndarray:
        """sample points in 2D with exponential decay from center"""

        scale_factor= self.params.get("scale_factor")
        if scale_factor is None:
            print("No cluster distribution params!")
            return None
        
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

            local_max_c=random.randint(1,self.params.get("max_c"))
            lambda_param=self.total_demand/num_centers/self.params.get("max_time")/(1+local_max_c/2)
            
            distribution=self.params.get("distribution")
            size=self.params.get("neighborhood_size",3)
            size=max(3,size)
            
            if distribution=="uniform":
                distribution_params={
                    "distribution":"uniform",
                    "size":random.uniform(0.75*size,1.25*size),
                    }
            elif distribution=="gaussian":
                distribution_params={
                    "distribution":"gaussian",
                    "sigma1":random.uniform(0.5*size,1.5*size),
                    "sigma2":random.uniform(0.5*size,1.5*size),
                    "rho":0.0,
                    }
            elif distribution=="cluster":
                distribution_params={
                    "distribution":"cluster",
                    "scale_factor":50*size,
                }

            local_params={
                "lambda_param":lambda_param,
                "max_c":local_max_c,
                "min_lifetime":self.params.get("min_lifetime",10),
                "max_lifetime":self.params.get("max_lifetime",25),
                "max_time":self.params.get("max_time",50),
                **distribution_params,
            }
           
            neighborhood = Neighborhood(
                center_x=center_x,
                center_y=center_y,
                width=self.width,
                height=self.height,
                rng=self._rng,
                params=local_params,
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
        for neighborhood in self.neighborhoods:
            demands = neighborhood.sample(t)
            all_demands.extend(demands)

        # Merge demands fallen into the same grid cell
        merged_demands = self._merge_demands_by_grid(all_demands)

        total_c = sum(d.c for d in merged_demands)
        self.total_demand -= total_c

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