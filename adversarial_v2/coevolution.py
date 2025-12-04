"""
Co-evolution Main Training Loop

Main training loop that alternates between:
1. Planner training: Using distributions from multiple generator versions
2. Generator training: Adversarial training to find planner weaknesses
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any
import json
import os
import random
import time
import torch

from .config import CoevolutionConfig
from .utils.registry import GeneratorRegistry
from .utils.training_visualizer import TrainingVisualizer
from .train_planner import PlannerTrainer
from .train_generator import GeneratorTrainer


def coevolution_loop(
    config: CoevolutionConfig,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main co-evolution training loop.
    
    Args:
        config: CoevolutionConfig object containing all settings
        resume_from: Path to checkpoint directory to resume from
        
    Returns:
        Dictionary with training history
    """
    print("=" * 60)
    print("Co-evolution Training")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Cycles: {config.num_cycles}")
    print(f"Planner epochs/cycle: {config.planner_epochs_per_cycle}")
    if config.first_cycle_planner_epochs is not None:
        print(f"First cycle planner epochs: {config.first_cycle_planner_epochs}")
    if config.planner_early_stop_patience:
        print(f"Planner early stop: patience={config.planner_early_stop_patience}, threshold={config.planner_early_stop_threshold}")
    print(f"Generator epochs/cycle: {config.generator_epochs_per_cycle}")
    if config.generator_early_stop_patience:
        print(f"Generator early stop: patience={config.generator_early_stop_patience}, threshold={config.generator_early_stop_threshold}")
    print(f"Version sampling policy: {config.version_sample_policy}")
    print(f"Map: {config.env.map_size}x{config.env.map_size}, Agents: {config.env.num_agents}")
    print(f"Num nodes: {config.env.num_nodes}, Total demand: {config.env.total_demand}")
    print(f"Save directory: {config.save_dir}")
    print("=" * 60)
    
    # Setup
    os.makedirs(config.save_dir, exist_ok=True)
    config_record_path = os.path.join(config.save_dir, "config_summary.json")
    with open(config_record_path, "w", encoding="utf-8") as cfg_file:
        json.dump(asdict(config), cfg_file, indent=2)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Initialize registry
    registry = GeneratorRegistry(save_dir=config.save_dir)
    
    # Initialize training visualizer
    visualizer = TrainingVisualizer(save_dir=config.save_dir)
    
    # Try to load existing registry and metrics
    if resume_from:
        registry.load()
        visualizer.load_metrics()
        print(f"Loaded registry with {registry.num_versions()} versions")
    
    # Initialize planner trainer (pass config object)
    planner_trainer = PlannerTrainer(config, registry, device)
    
    # Initialize generator trainer (pass config object)
    generator_trainer = GeneratorTrainer(config, planner_trainer.model, device)
    
    # Register initial generator version if registry is empty
    if registry.is_empty():
        init_gen_path = os.path.join(config.save_dir, "generator_v0.pth")
        torch.save(generator_trainer.get_state_dict(), init_gen_path)
        registry.add(init_gen_path, metrics={"cycle": 0, "tag": "init"})
        registry.save()
        print(f"Registered initial generator: {init_gen_path}")
    
    # Resume from checkpoint if specified
    start_cycle = 1
    if resume_from:
        # Look for latest planner checkpoint
        planner_ckpts = [f for f in os.listdir(config.save_dir) if f.startswith("planner_cycle_")]
        if planner_ckpts:
            latest_planner = max(planner_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            planner_path = os.path.join(config.save_dir, latest_planner)
            start_cycle = planner_trainer.load_checkpoint(planner_path) + 1
            print(f"Resumed planner from {planner_path}, starting cycle {start_cycle}")
        
        # Load latest generator into trainer
        latest_gen = registry.latest()
        if latest_gen:
            generator_trainer.load_checkpoint(latest_gen.checkpoint_path)
            print(f"Loaded latest generator: {latest_gen.checkpoint_path}")
    
    # Training history
    history = {
        "planner_scores": [],
        "planner_losses": [],
        "generator_rewards": [],
        "generator_losses": [],
    }
    
    # Main training loop
    for cycle in range(start_cycle, config.num_cycles + 1):
        print(f"\n{'='*60}")
        print(f"Cycle {cycle}/{config.num_cycles}")
        print(f"{'='*60}")
        
        # ============================================
        # Phase 1: Planner Training
        # ============================================
        print(f"\n[Phase 1] Training Planner (using {registry.num_versions()} generator versions)")
        
        # Use first_cycle_planner_epochs for cycle 1 if specified
        if cycle == 1 and config.first_cycle_planner_epochs is not None:
            planner_epochs = config.first_cycle_planner_epochs
            print(f"  Using first cycle epochs: {planner_epochs}")
        else:
            planner_epochs = config.planner_epochs_per_cycle
        
        # Early stopping state for this cycle
        best_score_in_cycle = float('inf')
        epochs_without_improvement = 0
        early_stop_patience = config.planner_early_stop_patience or 0
        early_stop_threshold = config.planner_early_stop_threshold
        
        for epoch in range(1, planner_epochs + 1):
            print(f"\n  Planner Epoch {epoch}/{planner_epochs}")
            epoch_start = time.time()
            
            metrics = planner_trainer.train_epoch()
            epoch_duration = time.time() - epoch_start
            
            current_score = metrics['score']
            print(f"  -> Score: {current_score:.4f}, Loss: {metrics['loss']:.4f}")
            if metrics.get("version_counts"):
                print(f"  -> Version usage: {metrics['version_counts']}")
            
            print(f"  -> Epoch duration: {epoch_duration:.2f}s")
            
            # Record to visualizer
            visualizer.add_planner_epoch(metrics["loss"], current_score)
            
            history["planner_scores"].append(current_score)
            history["planner_losses"].append(metrics["loss"])
            
            # Early stopping check (score is tour length, lower is better)
            if early_stop_patience > 0:
                improvement = best_score_in_cycle - current_score
                if improvement > early_stop_threshold:
                    # Improved
                    best_score_in_cycle = current_score
                    epochs_without_improvement = 0
                    print(f"  -> [Early Stop] New best score: {best_score_in_cycle:.4f}")
                else:
                    # No significant improvement
                    epochs_without_improvement += 1
                    print(f"  -> [Early Stop] No improvement for {epochs_without_improvement}/{early_stop_patience} epochs")
                    
                    if epochs_without_improvement >= early_stop_patience:
                        print(f"  -> [Early Stop] Stopping planner training early at epoch {epoch}/{planner_epochs}")
                        break
        
        # Save planner checkpoint
        planner_ckpt_path = os.path.join(config.save_dir, f"planner_cycle_{cycle}.pt")
        planner_trainer.save_checkpoint(planner_ckpt_path, epoch=cycle)
        
        # ============================================
        # Phase 2: Generator Training (Adversarial)
        # ============================================
        print(f"\n[Phase 2] Training Generator (adversarial against current planner)")
        
        # Update generator's planner reference
        generator_trainer.update_planner(planner_trainer.model)
        
        # Generator early stopping state for this cycle
        best_gen_reward_in_cycle = -float('inf')
        gen_epochs_without_improvement = 0
        gen_early_stop_patience = config.generator_early_stop_patience or 0
        gen_early_stop_threshold = config.generator_early_stop_threshold
        
        for epoch in range(1, config.generator_epochs_per_cycle + 1):
            print(f"\n  Generator Epoch {epoch}/{config.generator_epochs_per_cycle}")
            epoch_start = time.time()
            
            metrics = generator_trainer.train_epoch()
            epoch_duration = time.time() - epoch_start
            
            current_gen_reward = metrics['gen_reward']
            print(f"  -> Planner reward: {metrics['planner_reward']:.2f}, "
                  f"Gen reward: {current_gen_reward:.2f}, "
                  f"Loss: {metrics['loss']:.4f}")
            print(f"  -> Epoch duration: {epoch_duration:.2f}s")
            
            # Record to visualizer
            visualizer.add_generator_epoch(
                metrics["loss"], 
                metrics["gen_reward"], 
                metrics["planner_reward"]
            )
            
            history["generator_rewards"].append(metrics["gen_reward"])
            history["generator_losses"].append(metrics["loss"])
            
            # Generator early stopping check (gen_reward is higher = better)
            if gen_early_stop_patience > 0:
                improvement = current_gen_reward - best_gen_reward_in_cycle
                if improvement > gen_early_stop_threshold:
                    # Improved
                    best_gen_reward_in_cycle = current_gen_reward
                    gen_epochs_without_improvement = 0
                    print(f"  -> [Early Stop] New best gen_reward: {best_gen_reward_in_cycle:.4f}")
                    # Save best generator for this cycle
                    best_gen_path = os.path.join(config.save_dir, f"generator_cycle_{cycle}_best.pth")
                    generator_trainer.save_checkpoint(best_gen_path, epoch=cycle)
                else:
                    # No significant improvement
                    gen_epochs_without_improvement += 1
                    print(f"  -> [Early Stop] No improvement for {gen_epochs_without_improvement}/{gen_early_stop_patience} epochs")
                    
                    if gen_epochs_without_improvement >= gen_early_stop_patience:
                        print(f"  -> [Early Stop] Stopping generator training early at epoch {epoch}/{config.generator_epochs_per_cycle}")
                        break
        
        # Save generator checkpoint
        gen_ckpt_path = os.path.join(config.save_dir, f"generator_cycle_{cycle}.pth")
        generator_trainer.save_checkpoint(gen_ckpt_path, epoch=cycle)
        
        # Register new generator version
        registry.add(gen_ckpt_path, metrics={
            "cycle": cycle,
            "planner_reward": metrics["planner_reward"],
            "gen_reward": metrics["gen_reward"],
        })
        registry.save()
        
        # Update visualizer and save plots
        visualizer.end_cycle(cycle)
        visualizer.save_metrics()
        
        print(f"\n{registry.summary()}")
        
        # Update planner trainer's diffusion model with latest generator
        planner_trainer.load_generator_version(registry.latest())
    
    print("\n" + "=" * 60)
    print("Co-evolution Training Complete!")
    print("=" * 60)
    print(f"Final registry: {registry.num_versions()} generator versions")
    print(f"Checkpoints saved to: {config.save_dir}")
    
    return history
