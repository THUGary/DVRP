"""
Visualization tool for Diffusion Generator distribution.
Generates heatmaps showing the probability distribution of generated demands.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import prepare_condition, unnormalize_value, CONDITION_DIM


def load_diffusion_model(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> DemandDiffusionModel:
    """
    Load a trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded DemandDiffusionModel
    """
    model = DemandDiffusionModel(condition_dim=CONDITION_DIM).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def sample_demands(
    model: DemandDiffusionModel,
    num_samples: int = 1000,
    num_demands_per_sample: int = 50,
    map_size: Tuple[int, int] = (40, 40),
    total_demand: int = 60,
    max_c: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Sample demands from the diffusion model.
    
    Args:
        model: Trained diffusion model
        num_samples: Number of sampling iterations
        num_demands_per_sample: Number of demands per sample
        map_size: Grid size for the map (width, height)
        total_demand: Total demand for conditional generation
        max_c: Maximum capacity per node
        device: Device for computation
        
    Returns:
        Array of shape (total_demands, 5) with [t, x, y, c, lifetime]
    """
    condition = prepare_condition(total_demand=total_demand, max_c=max_c).to(device)
    
    all_demands = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample from diffusion model
            samples = model.sample(
                condition=condition,
                num_demands=num_demands_per_sample,
                grid_size=map_size
            )  # (num_demands, 5): [t_norm, x_norm, y_norm, c_norm, lifetime_norm]
            
            all_demands.append(samples.cpu().numpy())
    
    return np.concatenate(all_demands, axis=0)


def normalize_samples(demands: np.ndarray, max_time: float = 100.0, max_c: float = 5.0) -> np.ndarray:
    """
    Normalize raw model outputs to proper ranges.
    
    The diffusion model outputs values that should be in [-1, 1] range,
    but may not be if the model is not well trained. This function
    handles both cases.
    
    Args:
        demands: Raw model output (N, 5) with [t, x, y, c, lifetime]
        max_time: Maximum time value
        max_c: Maximum capacity value
        
    Returns:
        Normalized demands with proper ranges:
        - t: [0, max_time]
        - x, y: [0, 1]
        - c: [1, max_c]
        - lifetime: [0, max_time]
    """
    demands = demands.copy()
    
    # Check if values are in [-1, 1] range (well-trained model)
    # or need robust normalization (poorly trained / initial model)
    value_range = np.abs(demands).max()
    
    if value_range > 2.0:
        # Model outputs are not normalized - the model may not be well-trained
        # Use a fixed mapping assuming outputs are roughly centered around 0
        # and have some spread. We'll use percentile-based normalization
        # to handle outliers better.
        print(f"Warning: Model outputs not in [-1,1] range (max={value_range:.2f}), applying robust normalization")
        
        for col in range(demands.shape[1]):
            # Use percentile-based normalization to handle outliers
            p_low, p_high = np.percentile(demands[:, col], [2, 98])
            if p_high > p_low:
                # Map [p_low, p_high] to [0, 1]
                demands[:, col] = (demands[:, col] - p_low) / (p_high - p_low)
            else:
                demands[:, col] = 0.5
            # Clip to [0, 1]
            demands[:, col] = np.clip(demands[:, col], 0, 1)
        
        # Now map to proper ranges
        # t: [0, max_time], x/y: already [0,1], c: [1, max_c], lifetime: [0, max_time]
        demands[:, 0] = demands[:, 0] * max_time  # t
        # x, y already in [0, 1]
        demands[:, 3] = demands[:, 3] * (max_c - 1) + 1  # c: [1, max_c]
        demands[:, 4] = demands[:, 4] * max_time  # lifetime
    else:
        # Model outputs are in [-1, 1] range - use unnormalize_value
        demands[:, 0] = unnormalize_value(demands[:, 0], 0, max_time)  # t
        demands[:, 1] = unnormalize_value(demands[:, 1], 0, 1)  # x
        demands[:, 2] = unnormalize_value(demands[:, 2], 0, 1)  # y
        demands[:, 3] = unnormalize_value(demands[:, 3], 1, max_c)  # c
        demands[:, 4] = unnormalize_value(demands[:, 4], 0, max_time)  # lifetime
    
    # Clip to valid ranges
    demands[:, 0] = np.clip(demands[:, 0], 0, max_time)
    demands[:, 1] = np.clip(demands[:, 1], 0, 1)
    demands[:, 2] = np.clip(demands[:, 2], 0, 1)
    demands[:, 3] = np.clip(demands[:, 3], 1, max_c)
    demands[:, 4] = np.clip(demands[:, 4], 0, max_time)
    
    return demands


def create_heatmap(
    demands: np.ndarray,
    grid_size: int = 50,
    x_range: Tuple[float, float] = (0, 1),
    y_range: Tuple[float, float] = (0, 1),
    already_normalized: bool = True
) -> np.ndarray:
    """
    Create a 2D heatmap from demand positions.
    
    Args:
        demands: Array of shape (N, 5) with [t, x, y, c, lifetime]
        grid_size: Number of grid cells per dimension
        x_range: Range of x coordinates
        y_range: Range of y coordinates
        already_normalized: If True, x/y are already in [0, 1] range
        
    Returns:
        2D heatmap array of shape (grid_size, grid_size)
    """
    # Extract x, y coordinates (indices 1 and 2)
    x = demands[:, 1]
    y = demands[:, 2]
    
    # Clip to valid range
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])
    
    # Create 2D histogram
    heatmap, _, _ = np.histogram2d(
        x, y,
        bins=grid_size,
        range=[x_range, y_range]
    )
    
    # Normalize to probability
    heatmap = heatmap / heatmap.sum()
    
    return heatmap


def visualize_distribution(
    checkpoint_path: str,
    num_samples: int = 100,
    num_demands_per_sample: int = 50,
    grid_size: int = 50,
    map_size: Tuple[int, int] = (40, 40),
    total_demand: int = 60,
    max_c: int = 5,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "hot"
) -> np.ndarray:
    """
    Visualize the spatial distribution of a diffusion generator using a heatmap.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of sampling iterations
        num_demands_per_sample: Demands generated per iteration
        grid_size: Resolution of the heatmap
        map_size: Grid size for the map (width, height)
        total_demand: Total demand for conditional generation
        max_c: Maximum capacity per node
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
        cmap: Colormap for heatmap
        
    Returns:
        Heatmap array
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_diffusion_model(checkpoint_path, device)
    
    # Sample demands
    print(f"Sampling {num_samples * num_demands_per_sample} demands...")
    print(f"  Condition: total_demand={total_demand}, max_c={max_c}")
    demands = sample_demands(
        model, num_samples, num_demands_per_sample, map_size, 
        total_demand=total_demand, max_c=max_c, device=device
    )
    
    # Normalize samples to proper ranges
    max_time = 100.0
    demands = normalize_samples(demands, max_time=max_time, max_c=max_c)
    
    # Create heatmap
    print("Creating heatmap...")
    heatmap = create_heatmap(demands, grid_size)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Main heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(
        heatmap.T, 
        origin='lower', 
        cmap=cmap,
        extent=[0, 1, 0, 1],
        aspect='equal'
    )
    ax1.set_title('Spatial Distribution Heatmap', fontsize=14)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    plt.colorbar(im1, ax=ax1, label='Probability')
    
    # Log-scale heatmap for better visibility
    ax2 = axes[0, 1]
    log_heatmap = np.log10(heatmap + 1e-10)
    im2 = ax2.imshow(
        log_heatmap.T,
        origin='lower',
        cmap=cmap,
        extent=[0, 1, 0, 1],
        aspect='equal'
    )
    ax2.set_title('Log-scale Distribution', fontsize=14)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    plt.colorbar(im2, ax=ax2, label='Log10(Probability)')
    
    # Time distribution histogram (already normalized)
    ax3 = axes[1, 0]
    t_values = demands[:, 0]  # Already normalized to [0, max_time]
    ax3.hist(t_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_title('Time Distribution', fontsize=14)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Count')
    ax3.set_xlim(0, max_time)
    
    # Demand capacity distribution (already normalized)
    ax4 = axes[1, 1]
    c_values = demands[:, 3]  # Already normalized to [1, max_c]
    ax4.hist(c_values, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax4.set_title('Demand Capacity Distribution', fontsize=14)
    ax4.set_xlabel('Capacity')
    ax4.set_ylabel('Count')
    ax4.set_xlim(1, max_c)
    
    fig.suptitle(
        f'Diffusion Generator Distribution Analysis\n'
        f'Condition: total_demand={total_demand}, max_c={max_c}',
        fontsize=16, y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return heatmap


def visualize_episode_structure(
    checkpoint_path: str,
    num_demands: int = 50,
    map_size: Tuple[int, int] = (40, 40),
    total_demand: int = 60,
    max_c: int = 5,
    depot: Tuple[float, float] = (0.5, 0.5),
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 12),
) -> np.ndarray:
    """
    Visualize the structure of a single episode from the diffusion generator.
    
    Shows:
    - Scatter plot of demand positions with capacity as size
    - Density analysis (sparse vs crowded regions)
    - Center vs edge distribution
    - Time evolution of demands
    - Clustering analysis
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_demands: Number of demands to generate
        map_size: Grid size for the map
        total_demand: Total demand for conditional generation
        max_c: Maximum capacity per node
        depot: Depot position (normalized 0-1)
        save_path: Path to save figure
        show: Whether to display
        figsize: Figure size
        
    Returns:
        Generated demands array
    """
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and sample ONE episode
    model = load_diffusion_model(checkpoint_path, device)
    condition = prepare_condition(total_demand=total_demand, max_c=max_c).to(device)
    
    print(f"Generating {num_demands} demands for one episode...")
    print(f"  Condition: total_demand={total_demand}, max_c={max_c}")
    with torch.no_grad():
        samples = model.sample(
            condition=condition,
            num_demands=num_demands,
            grid_size=map_size
        )
    
    demands = samples.cpu().numpy()
    max_time = 100.0
    demands = normalize_samples(demands, max_time=max_time, max_c=max_c)
    
    # Extract columns
    t = demands[:, 0]  # time
    x = demands[:, 1]  # x position [0, 1]
    y = demands[:, 2]  # y position [0, 1]
    c = demands[:, 3]  # capacity
    lifetime = demands[:, 4]  # lifetime
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # ========== 1. Scatter plot with capacity as size ==========
    ax1 = fig.add_subplot(2, 3, 1)
    scatter = ax1.scatter(x, y, c=t, s=c*30, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.plot(depot[0], depot[1], 'r*', markersize=20, label='Depot', zorder=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Demand Positions\n(color=time, size=capacity)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    plt.colorbar(scatter, ax=ax1, label='Time Step')
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Density heatmap with KDE ==========
    ax2 = fig.add_subplot(2, 3, 2)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[0, 1], [0, 1]])
    # Smooth with Gaussian filter
    heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=1.0)
    im2 = ax2.imshow(heatmap_smooth.T, origin='lower', extent=[0, 1, 0, 1], 
                      cmap='YlOrRd', aspect='equal')
    ax2.plot(depot[0], depot[1], 'b*', markersize=15, label='Depot')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Density Heatmap\n(Sparse vs Crowded)', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Demand Count')
    ax2.legend(loc='upper right')
    
    # ========== 3. Distance from center analysis ==========
    ax3 = fig.add_subplot(2, 3, 3)
    center = np.array([0.5, 0.5])
    distances_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    distances_to_depot = np.sqrt((x - depot[0])**2 + (y - depot[1])**2)
    
    ax3.hist(distances_to_center, bins=20, alpha=0.6, label='From Center', color='blue', edgecolor='black')
    ax3.hist(distances_to_depot, bins=20, alpha=0.6, label='From Depot', color='red', edgecolor='black')
    ax3.axvline(x=0.5, color='green', linestyle='--', label='Map Diagonal/2')
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Count')
    ax3.set_title('Distance Distribution\n(Center vs Edge)', fontsize=12)
    ax3.legend()
    
    # Calculate metrics
    center_ratio = np.mean(distances_to_center < 0.3)  # Within 30% of center
    edge_ratio = np.mean(distances_to_center > 0.4)    # Outer 40%
    
    ax3.text(0.95, 0.95, f'Center (<0.3): {center_ratio:.1%}\nEdge (>0.4): {edge_ratio:.1%}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== 4. Time evolution ==========
    ax4 = fig.add_subplot(2, 3, 4)
    # Sort by time and show evolution
    sorted_idx = np.argsort(t)
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    
    for i, idx in enumerate(sorted_idx):
        ax4.scatter(x[idx], y[idx], c=[colors[i]], s=c[idx]*30, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    # Draw arrows showing time progression (first 10 points)
    if len(sorted_idx) > 1:
        for i in range(min(10, len(sorted_idx)-1)):
            idx1, idx2 = sorted_idx[i], sorted_idx[i+1]
            ax4.annotate('', xy=(x[idx2], y[idx2]), xytext=(x[idx1], y[idx1]),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3))
    
    ax4.plot(depot[0], depot[1], 'r*', markersize=15)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    ax4.set_title('Time Evolution\n(early=dark, late=light)', fontsize=12)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. Quadrant analysis ==========
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Divide into quadrants
    quadrants = np.zeros(4)
    quadrant_names = ['Bottom-Left', 'Bottom-Right', 'Top-Left', 'Top-Right']
    for xi, yi in zip(x, y):
        q_idx = int(xi >= 0.5) + 2 * int(yi >= 0.5)
        quadrants[q_idx] += 1
    
    bars = ax5.bar(quadrant_names, quadrants, color=['#ff9999', '#99ccff', '#99ff99', '#ffcc99'], edgecolor='black')
    ax5.axhline(y=len(x)/4, color='red', linestyle='--', label=f'Uniform: {len(x)/4:.1f}')
    ax5.set_ylabel('Demand Count')
    ax5.set_title('Quadrant Distribution\n(Balance Analysis)', fontsize=12)
    ax5.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars, quadrants):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    # ========== 6. Statistics Summary ==========
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate clustering (simple nearest neighbor)
    if len(x) > 1:
        positions = np.column_stack([x, y])
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_distances = dist_matrix.min(axis=1)
        avg_nn_dist = np.mean(nn_distances)
        clustering_score = 1.0 / (avg_nn_dist + 0.01)  # Higher = more clustered
    else:
        avg_nn_dist = 0
        clustering_score = 0
    
    # Summary statistics
    stats_text = f"""
    ===============================================
              EPISODE STRUCTURE ANALYSIS
    ===============================================
    
    [SPATIAL DISTRIBUTION]
       * Total demands: {len(x)}
       * X range: [{x.min():.3f}, {x.max():.3f}]
       * Y range: [{y.min():.3f}, {y.max():.3f}]
       * X std: {x.std():.3f}, Y std: {y.std():.3f}
    
    [CENTER vs EDGE]
       * Near center (<0.3): {center_ratio:.1%}
       * At edges (>0.4): {edge_ratio:.1%}
       * Avg distance to center: {distances_to_center.mean():.3f}
       * Avg distance to depot: {distances_to_depot.mean():.3f}
    
    [CLUSTERING]
       * Avg nearest neighbor dist: {avg_nn_dist:.4f}
       * Clustering score: {clustering_score:.2f}
       * (Higher = more clustered)
    
    [TEMPORAL]
       * Time range: [{t.min():.1f}, {t.max():.1f}]
       * Time std: {t.std():.1f}
    
    [CAPACITY]
       * Total capacity: {c.sum():.0f}
       * Avg capacity: {c.mean():.2f}
       * Max capacity: {c.max():.1f}
    
    ===============================================
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Single Episode Structure Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return demands


def compare_distributions(
    checkpoint_path: str,
    num_samples: int = 50,
    num_demands_per_sample: int = 50,
    grid_size: int = 40,
    map_size: Tuple[int, int] = (40, 40),
    total_demand: int = 60,
    max_c: int = 5,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate heatmap for diffusion model distribution.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Samples per distribution
        num_demands_per_sample: Demands per sample
        grid_size: Heatmap resolution
        map_size: Grid size for the map (width, height)
        total_demand: Total demand for conditional generation
        max_c: Maximum capacity per node
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Dictionary mapping distribution type to heatmap
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_diffusion_model(checkpoint_path, device)
    
    heatmaps = {}
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    print(f"Sampling with condition: total_demand={total_demand}, max_c={max_c}")
    demands = sample_demands(
        model, num_samples, num_demands_per_sample, map_size,
        total_demand=total_demand, max_c=max_c, device=device
    )
    
    # Normalize samples
    demands = normalize_samples(demands, max_time=100.0, max_c=max_c)
    
    heatmap = create_heatmap(demands, grid_size)
    heatmaps["diffusion"] = heatmap
    
    im = ax.imshow(
        heatmap.T,
        origin='lower',
        cmap='hot',
        extent=[0, 1, 0, 1],
        aspect='equal'
    )
    ax.set_title(f'Diffusion Distribution\n(total_demand={total_demand}, max_c={max_c})', fontsize=12)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.colorbar(im, ax=ax, label='Probability')
    
    fig.suptitle('Distribution Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    
    return heatmaps


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize diffusion generator distribution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--distribution", type=str, default="uniform", 
                        choices=["uniform", "gaussian", "cluster"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_demands", type=int, default=50)
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--map_size", type=int, default=40, help="Map size (square grid)")
    parser.add_argument("--total-demand", type=int, default=60, help="Total demand")
    parser.add_argument("--max-c", type=int, default=5, help="Max capacity")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="Compare all distribution types")
    parser.add_argument("--episode", action="store_true", help="Visualize single episode structure")
    parser.add_argument("--no_show", action="store_true", help="Don't display the plot")
    
    args = parser.parse_args()
    
    map_size = (args.map_size, args.map_size)
    show = not args.no_show
    total_demand = getattr(args, 'total_demand', 60)
    max_c = getattr(args, 'max_c', 5)
    
    if args.episode:
        visualize_episode_structure(
            checkpoint_path=args.checkpoint,
            num_demands=args.num_demands,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            save_path=args.save_path,
            show=show
        )
    elif args.compare:
        compare_distributions(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            num_demands_per_sample=args.num_demands,
            grid_size=args.grid_size,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            save_path=args.save_path,
            show=show
        )
    else:
        visualize_distribution(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            num_demands_per_sample=args.num_demands,
            grid_size=args.grid_size,
            map_size=map_size,
            total_demand=total_demand,
            max_c=max_c,
            save_path=args.save_path,
            show=show
        )
