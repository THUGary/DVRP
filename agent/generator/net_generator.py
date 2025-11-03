from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np

from .base import BaseDemandGenerator, Demand
from models.generator_model.diffusion_model import DemandDiffusionModel
# Import the new utility functions
from .data_utils import prepare_condition, unnormalize_value, CONDITION_DIM

class NetDemandGenerator(BaseDemandGenerator):
    """
    A demand generator that uses a trained neural network (e.g., a diffusion model)
    to generate all demands for an episode during the reset phase.
    """

    def __init__(self, width: int, height: int, **params: Any) -> None:
        super().__init__(width, height, **params)
        self._model: DemandDiffusionModel | None = None
        # This will store all demands for the episode, keyed by time step.
        self.pre_generated_demands: Dict[int, List[Demand]] = {}

    def reset(self, seed: int | None = None) -> None:
        """
        Loads the model and generates all demands for the entire episode.
        """
        # Use the provided seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load the trained model (if not already loaded)
        if self._model is None:
            self._model = DemandDiffusionModel(condition_dim=CONDITION_DIM)
            model_path = self.params.get("model_path", "checkpoints/diffusion_model.pth")
            try:
                # Load the trained model weights
                self._model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"NetDemandGenerator: Loaded trained model weights from {model_path}.")
            except FileNotFoundError:
                print(f"WARNING: Model weights not found at {model_path}. Using randomly initialized model.")
            self._model.to(device)
            self._model.eval()

        # 2. Prepare the conditional input for the model from generator_params
        # We need to add 'param_' prefix to match the CSV headers for the utility function
        params_for_condition = {f"param_{k}": v for k, v in self.params.items()}
        condition = prepare_condition(params_for_condition).unsqueeze(0).to(device)
        
        num_demands_to_generate = int(self.params.get("total_demand", 50))

        # 3. Run the model to generate all demands for the episode
        with torch.no_grad():
            # The model outputs a normalized tensor
            generated_demands_normalized = self._model.sample(
                condition, 
                num_demands_to_generate,
                grid_size=(self.width, self.height)
            )
        # ensure on cpu for numpy conversion
        generated_demands_normalized = generated_demands_normalized.cpu()
        # 4. Un-normalize, process, and store the model's output
        self._process_and_store_demands(generated_demands_normalized)
        print(f"NetDemandGenerator: Pre-generated all demands for the episode.")

    def sample(self, t: int) -> List[Demand]:
        """
        Returns the pre-generated demands for the given time step `t`.
        This is now a fast dictionary lookup.
        """
        return self.pre_generated_demands.get(t, [])

    def _process_and_store_demands(self, tensor: torch.Tensor) -> None:
        """
        Un-normalizes, validates, and converts the raw tensor output from the model 
        into a dictionary of Demand objects.
        """
        self.pre_generated_demands.clear()
        demands_np = tensor.cpu().numpy()

        # Get min/max values from params for un-normalization.
        # This will now raise a KeyError if a parameter is missing, which is the desired behavior.
        max_time = self.params["max_time"]
        max_c = self.params["max_c"]
        min_lifetime = self.params["min_lifetime"]
        max_lifetime = self.params["max_lifetime"]

        depot_xy: Optional[Tuple[int, int]] = tuple(self.depot) if getattr(self, "depot", None) is not None else None

        for demand_data in demands_np:
            # Use the utility function to un-normalize
            t_unnorm = unnormalize_value(demand_data[0], 0, max_time - 1)
            x_unnorm = unnormalize_value(demand_data[1], 0, self.width - 1)
            y_unnorm = unnormalize_value(demand_data[2], 0, self.height - 1)
            c_unnorm = unnormalize_value(demand_data[3], 1, max_c)
            lifetime_unnorm = unnormalize_value(demand_data[4], min_lifetime, max_lifetime)

            # Convert to integer and clip to valid ranges
            t = np.clip(int(round(t_unnorm)), 0, max_time - 1)
            x = np.clip(int(round(x_unnorm)), 0, self.width - 1)
            y = np.clip(int(round(y_unnorm)), 0, self.height - 1)
            c = np.clip(int(round(c_unnorm)), 1, max_c)
            lifetime = int(round(lifetime_unnorm))
            
            end_t = t + lifetime

            if depot_xy is not None and (x, y) == depot_xy:
                relocated = self._relocate_from_depot(x, y)
                if relocated is None:
                    continue
                x, y = relocated

            demand = Demand(x=x, y=y, t=t, c=c, end_t=end_t)

            # Store the demand in a dictionary keyed by its appearance time
            if t not in self.pre_generated_demands:
                self.pre_generated_demands[t] = []
            self.pre_generated_demands[t].append(demand)

    def _relocate_from_depot(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Find a nearby alternative cell when a demand collides with the depot."""
        depot = getattr(self, "depot", None)
        if depot is None:
            return (x, y)

        dx, dy = tuple(depot)
        if (x, y) != (dx, dy):
            return (x, y)

        max_radius = max(self.width, self.height)
        for radius in range(1, max_radius + 1):
            x_min = max(0, x - radius)
            x_max = min(self.width - 1, x + radius)
            y_min = max(0, y - radius)
            y_max = min(self.height - 1, y + radius)
            for ny in range(y_min, y_max + 1):
                for nx in range(x_min, x_max + 1):
                    if (nx, ny) == (dx, dy):
                        continue
                    return (nx, ny)

        for ny in range(self.height):
            for nx in range(self.width):
                if (nx, ny) != (dx, dy):
                    return (nx, ny)

        return None
