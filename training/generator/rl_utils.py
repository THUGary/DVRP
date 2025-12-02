import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # [NEW] Required for smoothing
from typing import List, Tuple, Dict, Any
from agent.generator.data_utils import unnormalize_value
from environment.env import GridEnvironment

def make_environment(cfg) -> GridEnvironment:
    """Creates the GridEnvironment from config."""
    env = GridEnvironment(
        width=cfg.width,
        height=cfg.height,
        num_agents=cfg.num_agents,
        capacity=cfg.capacity,
        depot=cfg.depot,
        max_time=cfg.max_time,
        expiry_penalty_scale=float(getattr(cfg, "expiry_penalty_scale", 5.0)),
        switch_penalty_scale=float(getattr(cfg, "switch_penalty_scale", 0.01)),
        capacity_reward_scale=float(getattr(cfg, "capacity_reward_scale", 10.0)),
        exploration_history_n=int(getattr(cfg, "exploration_history_n", 0)),
        exploration_penalty_scale=float(getattr(cfg, "exploration_penalty_scale", 0.0)),
        wait_penalty_scale=float(getattr(cfg, "wait_penalty_scale", 0.001)),
        max_end_time=int(getattr(cfg, "max_end_time", cfg.max_time * 2)),
        include_service_time=bool(getattr(cfg, "include_service_time", False)),
    )
    env.num_agents = cfg.num_agents
    return env

def apply_static_constraints(demands: List[Tuple[int,int,int,int,int]], max_time: int) -> List[Tuple[int,int,int,int,int]]:
    """Forces all demands to t=0 and extends deadline."""
    static_demands = []
    for (x, y, t, c, end_t) in demands:
        new_end_t = max_time + 1
        static_demands.append((x, y, 0, c, new_end_t))
    return static_demands

def calculate_spatial_entropy(demands: List[Tuple[int,int,int,int,int]]) -> float:
    """Calculates variance of x and y coordinates."""
    xs = [d[0] for d in demands]
    ys = [d[1] for d in demands]
    if len(xs) <= 1:
        return 0.0
    
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
    var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
    return var_x + var_y

def calculate_temporal_entropy(demands: List[Tuple[int,int,int,int,int]]) -> float:
    """Calculates variance of t (time)."""
    ts = [d[2] for d in demands]
    if len(ts) <= 1:
        return 0.0
    
    mean_t = sum(ts) / len(ts)
    var_t = sum((t - mean_t) ** 2 for t in ts) / len(ts)
    return var_t

def decode_demands_from_tensor(gen_tensor: torch.Tensor, params: Dict[str, Any]) -> List[Tuple[int,int,int,int,int]]:
    """
    Decodes the raw normalized output from the diffusion model into a list of demand tuples.
    Matches logic in NetDemandGenerator but returns simple tuples for training loop.
    """
    width = params["width"]
    height = params["height"]
    max_time = params["max_time"]
    max_c = params["max_c"]
    min_lifetime = params["min_lifetime"]
    max_lifetime = params["max_lifetime"]
    
    demands: List[Tuple[int,int,int,int,int]] = []
    for row in gen_tensor.cpu().numpy():
        t_raw, x_raw, y_raw, c_raw, life_raw = row
        t_val = int(round(unnormalize_value(t_raw, 0, max_time - 1)))
        x_val = int(round(unnormalize_value(x_raw, 0, width - 1)))
        y_val = int(round(unnormalize_value(y_raw, 0, height - 1)))
        c_val = int(round(unnormalize_value(c_raw, 1, max_c)))
        life_val = int(round(unnormalize_value(life_raw, min_lifetime, max_lifetime)))
        
        # Clip constraints
        t_val = max(0, min(max_time - 1, t_val))
        x_val = max(0, min(width - 1, x_val))
        y_val = max(0, min(height - 1, y_val))
        c_val = max(1, min(max_c, c_val))
        life_val = max(min_lifetime, min(max_lifetime, life_val))
        
        end_t = t_val + life_val
        demands.append((x_val, y_val, t_val, c_val, end_t))
    return demands

def normalize_demands_for_training(demands: List[Tuple[int,int,int,int,int]], cfg) -> torch.Tensor:
    """
    Re-normalizes demands into model space [0,1] or [-1,1] to compute noise-prediction loss.
    """
    dem_tensor = []
    max_time = cfg.max_time - 1
    width = cfg.width - 1
    height = cfg.height - 1
    max_c = cfg.generator_params['max_c']
    min_life = cfg.generator_params['min_lifetime']
    max_life = cfg.generator_params['max_lifetime']
    
    for (x,y,t,c,end_t) in demands:
        lifetime = end_t - t
        norm_t = (t - 0) / max_time
        norm_x = (x - 0) / width
        norm_y = (y - 0) / height
        norm_c = (c - 1) / (max_c - 1 if max_c > 1 else 1)
        norm_life = (lifetime - min_life) / (max_life - min_life if max_life > min_life else 1)
        dem_tensor.append([norm_t, norm_x, norm_y, norm_c, norm_life])
        
    if not dem_tensor:
        dem_tensor.append([0,0,0,0,0])
        
    return torch.tensor(dem_tensor, dtype=torch.float32)

def log_density_heatmap(writer, step, model, condition, cfg, device):
    """
    Generates a publication-quality continuous density heatmap.
    Logs to TensorBoard with history preservation.
    """
    model.eval()
    all_xs, all_ys = [], []
    
    # Sample multiple batches (e.g. 10) to estimate the true distribution
    # This aggregates ~500 points to reveal the "manifold" of the generator
    with torch.no_grad():
        for _ in range(10): 
            gen_tensor = model.sample(
                condition=condition, 
                num_demands=int(cfg.generator_params["total_demand"]), 
                grid_size=(cfg.width, cfg.height)
            )
            
            width = cfg.width
            height = cfg.height
            
            for row in gen_tensor.cpu().numpy():
                x_raw, y_raw = row[1], row[2]
                x_val = unnormalize_value(x_raw, 0, width - 1) 
                y_val = unnormalize_value(y_raw, 0, height - 1)
                all_xs.append(x_val)
                all_ys.append(y_val)

    # 1. Create a high-resolution grid for the heatmap
    bins_x = cfg.width
    bins_y = cfg.height
    heatmap, xedges, yedges = np.histogram2d(all_xs, all_ys, bins=[bins_x, bins_y], range=[[0, cfg.width], [0, cfg.height]])
    
    # 2. Apply Gaussian Smoothing (KDE approximation)
    # sigma=1.0 - 1.5 provides a nice organic look for 20x20 grids
    heatmap = gaussian_filter(heatmap.T, sigma=1.2) 

    # 3. Plotting
    # Use a dark background style if available, or default
    plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    
    # 'magma' is standard for AI papers. 
    # origin='lower' ensures (0,0) is bottom-left.
    im = ax.imshow(heatmap, origin='lower', cmap='magma', 
                   extent=[0, cfg.width, 0, cfg.height], 
                   interpolation='bicubic') # Bicubic makes it perfectly smooth
    
    # Add Depot
    ax.scatter([cfg.depot[0]], [cfg.depot[1]], c='cyan', marker='X', s=150, 
               label='Depot', edgecolors='white', linewidth=1.5, zorder=10)
    
    # Styling
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Probability Density', rotation=270, labelpad=15)
    
    ax.set_title(f"Generator Policy Density (Ep {step})", fontsize=12, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Remove grid lines for cleaner look
    ax.grid(False)

    # 4. Logging Strategy
    # Log to the main slider (Evolution)
    writer.add_figure("Policy/Evolution", fig, global_step=step)
    
    # Log to a unique tag to prevent overwriting (Snapshots)
    writer.add_figure(f"Policy_Snapshots/Ep_{step:05d}", fig, global_step=step)
    
    plt.close(fig)