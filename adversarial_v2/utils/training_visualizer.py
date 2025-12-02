"""
Training Visualizer for Co-evolution Training

Generates and updates training curves for:
- Loss (planner and generator)
- Reward (planner and generator)  
- Score (planner)

Updates plots at the end of each cycle, replacing old plots.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingMetrics:
    """Container for all training metrics."""
    # Planner metrics (per epoch)
    planner_losses: List[float] = field(default_factory=list)
    planner_scores: List[float] = field(default_factory=list)
    
    # Generator metrics (per epoch)
    generator_losses: List[float] = field(default_factory=list)
    generator_rewards: List[float] = field(default_factory=list)
    planner_rewards: List[float] = field(default_factory=list)  # From generator's perspective
    
    # Cycle boundaries (epoch indices where cycles end)
    cycle_boundaries: List[int] = field(default_factory=list)
    
    def add_planner_epoch(self, loss: float, score: float):
        """Record metrics from one planner epoch."""
        self.planner_losses.append(loss)
        self.planner_scores.append(score)
    
    def add_generator_epoch(self, loss: float, gen_reward: float, planner_reward: float):
        """Record metrics from one generator epoch."""
        self.generator_losses.append(loss)
        self.generator_rewards.append(gen_reward)
        self.planner_rewards.append(planner_reward)
    
    def mark_cycle_end(self):
        """Mark the end of a cycle for visualization."""
        # Record current epoch counts as cycle boundary
        planner_epoch = len(self.planner_losses)
        generator_epoch = len(self.generator_losses)
        self.cycle_boundaries.append((planner_epoch, generator_epoch))
    
    def get_cycle_count(self) -> int:
        """Get number of completed cycles."""
        return len(self.cycle_boundaries)


class TrainingVisualizer:
    """
    Visualizes training progress for co-evolution training.
    
    Creates three separate plots:
    1. Loss plot: Planner loss and Generator loss
    2. Reward plot: Generator reward and Planner reward (from gen perspective)
    3. Score plot: Planner score
    
    All plots show both planner and generator curves with cycle boundaries.
    """
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save plot images
        """
        self.save_dir = save_dir
        self.metrics = TrainingMetrics()
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot file paths
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
        """Update all three plots."""
        self._plot_losses(cycle)
        self._plot_rewards(cycle)
        self._plot_scores(cycle)
        print(f"  [Visualizer] Updated training plots (cycle {cycle})")
    
    def _plot_losses(self, cycle: int):
        """Plot planner and generator losses."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Planner losses
        planner_epochs = np.arange(1, len(self.metrics.planner_losses) + 1)
        if len(planner_epochs) > 0:
            ax.plot(planner_epochs, self.metrics.planner_losses, 
                   'b-', linewidth=2, label='Planner Loss', alpha=0.8)
            # Add smoothed line
            if len(self.metrics.planner_losses) > 5:
                smoothed = self._smooth(self.metrics.planner_losses)
                ax.plot(planner_epochs, smoothed, 'b--', linewidth=1, alpha=0.5)
        
        # Generator losses (on secondary x-axis conceptually, but we'll scale)
        gen_epochs = np.arange(1, len(self.metrics.generator_losses) + 1)
        if len(gen_epochs) > 0:
            # Scale generator epochs to align with planner epochs for visualization
            scale = len(planner_epochs) / max(len(gen_epochs), 1) if len(planner_epochs) > 0 else 1
            scaled_gen_epochs = gen_epochs * scale
            ax.plot(scaled_gen_epochs, self.metrics.generator_losses,
                   'r-', linewidth=2, label='Generator Loss', alpha=0.8)
            if len(self.metrics.generator_losses) > 5:
                smoothed = self._smooth(self.metrics.generator_losses)
                ax.plot(scaled_gen_epochs, smoothed, 'r--', linewidth=1, alpha=0.5)
        
        # Add cycle boundaries
        self._add_cycle_boundaries(ax, 'planner')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Loss (Cycle {cycle})', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rewards(self, cycle: int):
        """Plot generator and planner rewards."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        gen_epochs = np.arange(1, len(self.metrics.generator_rewards) + 1)
        
        if len(gen_epochs) > 0:
            # Generator reward (negative of planner performance)
            ax.plot(gen_epochs, self.metrics.generator_rewards,
                   'r-', linewidth=2, label='Generator Reward', alpha=0.8)
            
            # Planner reward (from generator's perspective)
            ax.plot(gen_epochs, self.metrics.planner_rewards,
                   'b-', linewidth=2, label='Planner Reward', alpha=0.8)
            
            # Add smoothed lines
            if len(self.metrics.generator_rewards) > 5:
                ax.plot(gen_epochs, self._smooth(self.metrics.generator_rewards),
                       'r--', linewidth=1, alpha=0.5)
                ax.plot(gen_epochs, self._smooth(self.metrics.planner_rewards),
                       'b--', linewidth=1, alpha=0.5)
            
            # Zero line
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Add cycle boundaries
        self._add_cycle_boundaries(ax, 'generator')
        
        ax.set_xlabel('Generator Epoch', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'Training Rewards (Cycle {cycle})', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reward_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_scores(self, cycle: int):
        """Plot planner scores."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        planner_epochs = np.arange(1, len(self.metrics.planner_scores) + 1)
        
        if len(planner_epochs) > 0:
            ax.plot(planner_epochs, self.metrics.planner_scores,
                   'g-', linewidth=2, label='Planner Score', alpha=0.8)
            
            # Add smoothed line
            if len(self.metrics.planner_scores) > 5:
                smoothed = self._smooth(self.metrics.planner_scores)
                ax.plot(planner_epochs, smoothed, 'g--', linewidth=1, alpha=0.5)
            
            # Add running max
            running_max = np.maximum.accumulate(self.metrics.planner_scores)
            ax.plot(planner_epochs, running_max, 'k:', linewidth=1, 
                   label='Best Score', alpha=0.7)
        
        # Add cycle boundaries
        self._add_cycle_boundaries(ax, 'planner')
        
        ax.set_xlabel('Planner Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Planner Score (Cycle {cycle})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.score_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _add_cycle_boundaries(self, ax, metric_type: str):
        """Add vertical lines at cycle boundaries."""
        for i, (planner_end, gen_end) in enumerate(self.metrics.cycle_boundaries):
            if metric_type == 'planner':
                x = planner_end
            else:
                x = gen_end
            
            if x > 0:
                ax.axvline(x=x, color='purple', linestyle='--', 
                          alpha=0.5, linewidth=1)
                ax.text(x, ax.get_ylim()[1], f'C{i+1}', 
                       fontsize=8, ha='center', va='bottom', color='purple')
    
    def _smooth(self, data: List[float], window: int = 5) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(data) < window:
            return np.array(data)
        
        kernel = np.ones(window) / window
        # Pad to handle edges
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
        
        self.metrics.planner_losses = data.get("planner_losses", [])
        self.metrics.planner_scores = data.get("planner_scores", [])
        self.metrics.generator_losses = data.get("generator_losses", [])
        self.metrics.generator_rewards = data.get("generator_rewards", [])
        self.metrics.planner_rewards = data.get("planner_rewards", [])
        self.metrics.cycle_boundaries = [tuple(x) for x in data.get("cycle_boundaries", [])]
        
        print(f"  [Visualizer] Loaded metrics from {path}")
        return True
