"""
Demand Converter

Convert diffusion model output to static or dynamic VRP demands.

Static mode: All demands appear at t=0, deadline=max_time
Dynamic mode: Demands appear at different times with varying deadlines
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch

# Import normalization utilities from project
from agent.generator.data_utils import unnormalize_value
from configs import DEMAND_NORM


@dataclass
class DemandTuple:
    """A single demand/customer."""
    x: int
    y: int
    t: int  # arrival time (0 for static)
    c: int  # demand capacity
    end_t: int  # deadline


class DemandConverter:
    """
    Converts diffusion model outputs to VRP demands.
    
    The diffusion model outputs normalized values in shape (num_demands, 5):
    [t_norm, x_norm, y_norm, c_norm, lifetime_norm]
    """
    
    def __init__(
        self,
        map_size: int = 20,
        max_time: int = 100,
        max_c: int = 5,
        min_lifetime: int = 10,
        max_lifetime: int = 50,
    ):
        self.map_size = map_size
        self.max_time = max_time
        self.max_c = max_c
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
    
    def convert_to_static(
        self, 
        diffusion_output: torch.Tensor,
    ) -> List[DemandTuple]:
        """
        Convert diffusion output to static VRP demands.
        
        All demands appear at t=0 and have deadline=max_time.
        Only x, y, c are used from the diffusion output.
        
        Args:
            diffusion_output: (num_demands, 5) tensor from diffusion model
            
        Returns:
            List of DemandTuple
        """
        demands = []
        
        for row in diffusion_output.cpu().numpy():
            # Diffusion output format: [t_norm, x_norm, y_norm, c_norm, lifetime_norm]
            _, x_norm, y_norm, c_norm, _ = row
            
            # Unnormalize x, y, c
            x = int(round(unnormalize_value(x_norm, 0, self.map_size - 1)))
            y = int(round(unnormalize_value(y_norm, 0, self.map_size - 1)))
            c = int(round(unnormalize_value(c_norm, 1, self.max_c)))
            
            # Clamp to valid ranges
            x = max(0, min(self.map_size - 1, x))
            y = max(0, min(self.map_size - 1, y))
            c = max(1, min(self.max_c, c))
            
            # Static: t=0, end_t=max_time
            demands.append(DemandTuple(x=x, y=y, t=0, c=c, end_t=self.max_time))
        
        return demands
    
    def convert_to_dynamic(
        self,
        diffusion_output: torch.Tensor,
    ) -> List[DemandTuple]:
        """
        Convert diffusion output to dynamic VRP demands.
        
        Uses all fields: t, x, y, c, lifetime to create time-varying demands.
        
        Args:
            diffusion_output: (num_demands, 5) tensor from diffusion model
            
        Returns:
            List of DemandTuple
        """
        demands = []
        
        for row in diffusion_output.cpu().numpy():
            t_norm, x_norm, y_norm, c_norm, life_norm = row
            
            # Unnormalize all values
            t = int(round(unnormalize_value(t_norm, 0, self.max_time - 1)))
            x = int(round(unnormalize_value(x_norm, 0, self.map_size - 1)))
            y = int(round(unnormalize_value(y_norm, 0, self.map_size - 1)))
            c = int(round(unnormalize_value(c_norm, 1, self.max_c)))
            lifetime = int(round(unnormalize_value(life_norm, self.min_lifetime, self.max_lifetime)))
            
            # Clamp to valid ranges
            t = max(0, min(self.max_time - 1, t))
            x = max(0, min(self.map_size - 1, x))
            y = max(0, min(self.map_size - 1, y))
            c = max(1, min(self.max_c, c))
            lifetime = max(self.min_lifetime, min(self.max_lifetime, lifetime))
            
            end_t = t + lifetime
            demands.append(DemandTuple(x=x, y=y, t=t, c=c, end_t=end_t))
        
        return demands
    
    def convert(
        self,
        diffusion_output: torch.Tensor,
        mode: str = "static",
    ) -> List[DemandTuple]:
        """
        Convert diffusion output to demands based on mode.
        
        Args:
            diffusion_output: (num_demands, 5) tensor from diffusion model
            mode: "static" or "dynamic"
            
        Returns:
            List of DemandTuple
        """
        if mode == "static":
            return self.convert_to_static(diffusion_output)
        elif mode == "dynamic":
            return self.convert_to_dynamic(diffusion_output)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'static' or 'dynamic'.")
    
    def to_tensor_batch(
        self,
        demands: List[DemandTuple],
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Convert demands to tensor batch format for V2Planner.
        
        Returns dict with:
            - depot_xy: (1, 1, 2) - Will be set externally
            - node_xy: (1, N, 2) - Normalized coordinates
            - node_demand: (1, N) - Normalized demands
            - node_deadline: (1, N) - For dynamic mode
        """
        N = len(demands)
        
        # Normalize coordinates to [0, 1]
        node_coords = [[d.x / (self.map_size - 1), d.y / (self.map_size - 1)] for d in demands]
        node_xy = torch.tensor([node_coords], dtype=torch.float32, device=device)
        
        # Normalize demands by DEMAND_NORM (capacity)
        node_demand = torch.tensor([[d.c / DEMAND_NORM for d in demands]], 
                                   dtype=torch.float32, device=device)
        
        # Deadlines (normalized by max_time)
        node_deadline = torch.tensor([[d.end_t / self.max_time for d in demands]], 
                                     dtype=torch.float32, device=device)
        
        return {
            "node_xy": node_xy,
            "node_demand": node_demand, 
            "node_deadline": node_deadline,
        }
    
    def to_raw_list(
        self,
        demands: List[DemandTuple],
    ) -> List[Tuple[int, int, int, int, int]]:
        """Convert to raw tuple list format for environment."""
        return [(d.x, d.y, d.t, d.c, d.end_t) for d in demands]


def generate_demands_from_diffusion(
    model: torch.nn.Module,
    condition: torch.Tensor,
    num_demands: int,
    map_size: int,
    mode: str = "static",
    converter: Optional[DemandConverter] = None,
    device: str = "cpu",
) -> List[DemandTuple]:
    """
    Generate demands using diffusion model.
    
    Args:
        model: DemandDiffusionModel
        condition: Condition tensor
        num_demands: Number of demands to generate
        map_size: Side length of the square map
        mode: "static" or "dynamic"
        converter: Optional DemandConverter instance
        device: Device to use
        
    Returns:
        List of DemandTuple
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from diffusion model
        output = model.sample(
            condition=condition.to(device),
            num_demands=num_demands,
            grid_size=(map_size, map_size),
        )
    
    # Convert using provided or default converter
    if converter is None:
        converter = DemandConverter(
            map_size=map_size,
        )
    
    return converter.convert(output, mode=mode)
