"""
Training Visualizer for Co-evolution Training

Generates and updates training curves for:
- Planner Loss
- Planner Score
- Generator Loss
- Generator Reward
- Planner Reward (from generator's perspective)

Each metric has its own plot. During phases where the metric is not being trained,
the plot shows gaps (NaN values) to clearly indicate training phases.

Updates plots at the end of each cycle, replacing old plots.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingMetrics:
    """Container for all training metrics with step tracking."""
    # Planner metrics (per epoch) - stored as (global_step, value)
    planner_losses: List[Tuple[int, float]] = field(default_factory=list)
    planner_scores: List[Tuple[int, float]] = field(default_factory=list)
    
    # Generator metrics (per epoch) - stored as (global_step, value)
    generator_losses: List[Tuple[int, float]] = field(default_factory=list)
    generator_rewards: List[Tuple[int, float]] = field(default_factory=list)
    planner_rewards: List[Tuple[int, float]] = field(default_factory=list)  # From generator's perspective
    
    # Global step counter
    global_step: int = 0
    
    # Cycle boundaries (global step where each cycle ends)
    cycle_boundaries: List[int] = field(default_factory=list)
    
    # Track phase transitions for visualization
    phase_transitions: List[Tuple[int, str]] = field(default_factory=list)  # (step, phase_name)
    
    def add_planner_epoch(self, loss: float, score: float):
        """Record metrics from one planner epoch."""
        self.global_step += 1
        self.planner_losses.append((self.global_step, loss))
        self.planner_scores.append((self.global_step, score))
    
    def add_generator_epoch(self, loss: float, gen_reward: float, planner_reward: float):
        """Record metrics from one generator epoch."""
        self.global_step += 1
        self.generator_losses.append((self.global_step, loss))
        self.generator_rewards.append((self.global_step, gen_reward))
        self.planner_rewards.append((self.global_step, planner_reward))
    
    def mark_cycle_end(self):
        """Mark the end of a cycle for visualization."""
        self.cycle_boundaries.append(self.global_step)
    
    def get_cycle_count(self) -> int:
        """Get number of completed cycles."""
        return len(self.cycle_boundaries)


class TrainingVisualizer:
    """
    Visualizes training progress for co-evolution training.
    
    Creates six separate plots (one for each metric):
    1. planner_loss.png - Planner training loss
    2. planner_score.png - Planner score
    3. generator_loss.png - Generator training loss
    4. generator_reward.png - Generator reward
    5. planner_reward.png - Planner reward (from generator's perspective)
    6. training_overview.png - Combined overview (2x3 grid)
    
    Each plot uses global step as x-axis. During phases where a metric
    is not being recorded, the plot shows gaps.
    """
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save plot images
        """
        self.save_dir = save_dir
        self.metrics = TrainingMetrics()
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot file paths - 6 individual plots + 1 overview
        self.plot_paths = {
            "planner_loss": os.path.join(save_dir, "planner_loss.png"),
            "planner_score": os.path.join(save_dir, "planner_score.png"),
            "generator_loss": os.path.join(save_dir, "generator_loss.png"),
            "generator_reward": os.path.join(save_dir, "generator_reward.png"),
            "planner_reward": os.path.join(save_dir, "planner_reward.png"),
            "overview": os.path.join(save_dir, "training_overview.png"),
        }
        
        # Legacy paths for backward compatibility
        self.loss_plot_path = os.path.join(save_dir, "training_loss.png")
        self.reward_plot_path = os.path.join(save_dir, "training_reward.png")
        self.score_plot_path = os.path.join(save_dir, "training_score.png")
    
    def add_planner_epoch(self, loss: float, score: float):
        """Record planner training metrics for one epoch."""
        self.metrics.add_planner_epoch(loss, score)
    
    def add_generator_epoch(self, loss: float, gen_reward: float, planner_reward: float):
        """Record generator training metrics for one epoch."""
        self.metrics.add_generator_epoch(loss, gen_reward, planner_reward)
    
    def end_cycle(self, cycle: int):
        """
        Mark end of a cycle and update all plots.
        
        Args:
            cycle: Current cycle number
        """
        self.metrics.mark_cycle_end()
        self._update_all_plots(cycle)
    
    def _update_all_plots(self, cycle: int):
        """Update all six individual plots and the overview."""
        self._plot_single_metric(
            data=self.metrics.planner_losses,
            title=f"Planner Loss (Cycle {cycle})",
            ylabel="Loss",
            color="blue",
            save_path=self.plot_paths["planner_loss"],
        )
        
        self._plot_single_metric(
            data=self.metrics.planner_scores,
            title=f"Planner Score (Cycle {cycle})",
            ylabel="Score",
            color="green",
            save_path=self.plot_paths["planner_score"],
            show_best=True,
        )
        
        self._plot_single_metric(
            data=self.metrics.generator_losses,
            title=f"Generator Loss (Cycle {cycle})",
            ylabel="Loss",
            color="red",
            save_path=self.plot_paths["generator_loss"],
        )
        
        self._plot_single_metric(
            data=self.metrics.generator_rewards,
            title=f"Generator Reward (Cycle {cycle})",
            ylabel="Reward",
            color="orange",
            save_path=self.plot_paths["generator_reward"],
            show_zero_line=True,
        )
        
        self._plot_single_metric(
            data=self.metrics.planner_rewards,
            title=f"Planner Reward (Cycle {cycle})",
            ylabel="Reward",
            color="purple",
            save_path=self.plot_paths["planner_reward"],
        )
        
        # Create overview plot
        self._plot_overview(cycle)
        
        print(f"  [Visualizer] Updated training plots (cycle {cycle})")
    
    def _plot_single_metric(
        self,
        data: List[Tuple[int, float]],
        title: str,
        ylabel: str,
        color: str,
        save_path: str,
        show_best: bool = False,
        show_zero_line: bool = False,
    ):
        """
        Plot a single metric with gaps where data doesn't exist.
        
        Args:
            data: List of (step, value) tuples
            title: Plot title
            ylabel: Y-axis label
            color: Line color
            save_path: Path to save the plot
            show_best: If True, show running best line
            show_zero_line: If True, show y=0 reference line
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if len(data) > 0:
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            
            # Plot main line
            ax.plot(steps, values, f'{color[0]}-', linewidth=2, label=ylabel, alpha=0.8)
            
            # Add smoothed line if enough data
            if len(values) > 5:
                smoothed = self._smooth(values)
                ax.plot(steps, smoothed, f'{color[0]}--', linewidth=1, alpha=0.5, label=f'{ylabel} (smoothed)')
            
            # Show running best for score
            if show_best and len(values) > 0:
                running_best = np.maximum.accumulate(values)
                ax.plot(steps, running_best, 'k:', linewidth=1, label='Best', alpha=0.7)
            
            # Show zero line for reward
            if show_zero_line:
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Add cycle boundaries
        self._add_cycle_boundaries(ax)
        
        # Styling
        ax.set_xlabel('Global Step', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to start from 0
        if self.metrics.global_step > 0:
            ax.set_xlim(0, self.metrics.global_step + 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_overview(self, cycle: int):
        """Create a 2x3 overview plot with all metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Training Overview (Cycle {cycle})', fontsize=14, fontweight='bold')
        
        # Plot configurations: (data, title, ylabel, color, ax_position, options)
        plot_configs = [
            (self.metrics.planner_losses, 'Planner Loss', 'Loss', 'blue', (0, 0), {}),
            (self.metrics.planner_scores, 'Planner Score', 'Score', 'green', (0, 1), {'show_best': True}),
            (self.metrics.generator_losses, 'Generator Loss', 'Loss', 'red', (0, 2), {}),
            (self.metrics.generator_rewards, 'Generator Reward', 'Reward', 'orange', (1, 0), {'show_zero_line': True}),
            (self.metrics.planner_rewards, 'Planner Reward', 'Reward', 'purple', (1, 1), {}),
        ]
        
        for data, title, ylabel, color, (row, col), options in plot_configs:
            ax = axes[row, col]
            
            if len(data) > 0:
                steps = [d[0] for d in data]
                values = [d[1] for d in data]
                
                ax.plot(steps, values, f'{color[0]}-', linewidth=1.5, alpha=0.8)
                
                if len(values) > 5:
                    smoothed = self._smooth(values)
                    ax.plot(steps, smoothed, f'{color[0]}--', linewidth=1, alpha=0.4)
                
                if options.get('show_best') and len(values) > 0:
                    running_best = np.maximum.accumulate(values)
                    ax.plot(steps, running_best, 'k:', linewidth=1, alpha=0.5)
                
                if options.get('show_zero_line'):
                    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            
            # Add cycle boundaries
            for boundary in self.metrics.cycle_boundaries:
                ax.axvline(x=boundary, color='purple', linestyle='--', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('Step', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if self.metrics.global_step > 0:
                ax.set_xlim(0, self.metrics.global_step + 1)
        
        # Hide the last subplot (2, 2) - use it for legend/info
        ax_info = axes[1, 2]
        ax_info.axis('off')
        
        # Add summary text
        summary_text = f"Cycle: {cycle}\n"
        summary_text += f"Total Steps: {self.metrics.global_step}\n\n"
        
        if len(self.metrics.planner_scores) > 0:
            latest_score = self.metrics.planner_scores[-1][1]
            best_score = max(v for _, v in self.metrics.planner_scores)
            summary_text += f"Planner Score:\n  Latest: {latest_score:.4f}\n  Best: {best_score:.4f}\n\n"
        
        if len(self.metrics.generator_rewards) > 0:
            latest_gen_reward = self.metrics.generator_rewards[-1][1]
            summary_text += f"Generator Reward:\n  Latest: {latest_gen_reward:.2f}\n"
        
        ax_info.text(0.1, 0.9, summary_text, transform=ax_info.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.plot_paths["overview"], dpi=150, bbox_inches='tight')
        plt.close()
    
    def _add_cycle_boundaries(self, ax):
        """Add vertical lines at cycle boundaries."""
        for i, boundary in enumerate(self.metrics.cycle_boundaries):
            ax.axvline(x=boundary, color='purple', linestyle='--', alpha=0.5, linewidth=1)
            # Add cycle label at top
            ylim = ax.get_ylim()
            ax.text(boundary, ylim[1], f'C{i+1}', fontsize=8, ha='center', va='bottom', 
                   color='purple', alpha=0.7)
    
    def _smooth(self, data: List[float], window: int = 5) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(data) < window:
            return np.array(data)
        
        kernel = np.ones(window) / window
        padded = np.pad(data, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(data)]
    
    def save_metrics(self, path: Optional[str] = None):
        """Save metrics to a file for later analysis."""
        import json
        
        if path is None:
            path = os.path.join(self.save_dir, "training_metrics.json")
        
        data = {
            "planner_losses": self.metrics.planner_losses,
            "planner_scores": self.metrics.planner_scores,
            "generator_losses": self.metrics.generator_losses,
            "generator_rewards": self.metrics.generator_rewards,
            "planner_rewards": self.metrics.planner_rewards,
            "global_step": self.metrics.global_step,
            "cycle_boundaries": self.metrics.cycle_boundaries,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  [Visualizer] Saved metrics to {path}")
    
    def load_metrics(self, path: Optional[str] = None):
        """Load metrics from a file."""
        import json
        
        if path is None:
            path = os.path.join(self.save_dir, "training_metrics.json")
        
        if not os.path.exists(path):
            return False
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (list of values) and new format (list of tuples)
        def convert_to_tuples(values, start_step=1):
            if len(values) == 0:
                return []
            if isinstance(values[0], (list, tuple)):
                return [tuple(v) for v in values]
            else:
                # Old format: convert to tuples
                return [(i + start_step, v) for i, v in enumerate(values)]
        
        self.metrics.planner_losses = convert_to_tuples(data.get("planner_losses", []))
        self.metrics.planner_scores = convert_to_tuples(data.get("planner_scores", []))
        self.metrics.generator_losses = convert_to_tuples(data.get("generator_losses", []))
        self.metrics.generator_rewards = convert_to_tuples(data.get("generator_rewards", []))
        self.metrics.planner_rewards = convert_to_tuples(data.get("planner_rewards", []))
        self.metrics.global_step = data.get("global_step", 0)
        self.metrics.cycle_boundaries = data.get("cycle_boundaries", [])
        
        # If global_step not saved, compute from data
        if self.metrics.global_step == 0:
            all_steps = []
            for metric_list in [self.metrics.planner_losses, self.metrics.planner_scores,
                               self.metrics.generator_losses, self.metrics.generator_rewards,
                               self.metrics.planner_rewards]:
                if metric_list:
                    all_steps.extend([s for s, _ in metric_list])
            if all_steps:
                self.metrics.global_step = max(all_steps)
        
        print(f"  [Visualizer] Loaded metrics from {path}")
        return True
