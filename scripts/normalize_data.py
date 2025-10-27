"""
This script normalizes the DVRP demand dataset by processing each episode in parallel. 
It reads raw demand data from a CSV file,
normalizes the demand features using predefined normalization functions, 
and saves the processed dataset into a PyTorch .pt file for efficient loading during training.
"""


import os
import sys
import torch
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import argparse

# Add project root to path to allow importing from other folders
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from agent.generator.data_utils import normalize_value, prepare_condition
from configs import get_default_config

# This is likely not needed anymore but is harmless to keep.
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass # Strategy already set

# Create a worker initializer to share config data efficiently ---
def init_worker(cfg):
    """Initializes worker process with global variables for map dimensions."""
    global worker_width, worker_height
    worker_width = cfg.width
    worker_height = cfg.height

def normalize_episode(episode_group):
    """
    Worker function to normalize all data for a single episode.
    Returns ONLY numpy arrays and basic types to avoid IPC issues.
    """
    episode_id, episode_df = episode_group
    
    params = episode_df.iloc[0].to_dict()
    
    # --- Convert condition tensor to numpy array immediately ---
    condition_tensor = prepare_condition(params)
    condition_np = condition_tensor.numpy()

    demands = episode_df[['time_step', 'demand_x', 'demand_y', 'demand_c', 'demand_end_t']].values
    
    max_time = params['param_max_time']
    max_c = params['param_max_c']
    min_lifetime = params['param_min_lifetime']
    max_lifetime = params['param_max_lifetime']
    
    # Normalize 5 elements of the demands
    demands_norm = np.zeros_like(demands, dtype=np.float32)
    demands_norm[:, 0] = normalize_value(demands[:, 0], 0, max_time - 1)
    # --- FIX: Use the globally set width and height from the config ---
    demands_norm[:, 1] = normalize_value(demands[:, 1], 0, worker_width - 1)
    demands_norm[:, 2] = normalize_value(demands[:, 2], 0, worker_height - 1)
    demands_norm[:, 3] = normalize_value(demands[:, 3], 1, max_c)
    lifetimes = demands[:, 4] - demands[:, 0]
    demands_norm[:, 4] = normalize_value(lifetimes, min_lifetime, max_lifetime)
    
    return {
        'id': episode_id,
        'condition': condition_np, # Return numpy array
        'demands': demands_norm    # Return numpy array
    }

def process_data_parallel(input_file, output_file):
    """
    Reads a raw dataset, normalizes it in parallel, and saves the result.
    """
    print(f"Processing raw data from: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    df = pd.read_csv(input_file)
    episode_ids = df['episode_id'].unique()
    
    # --- FIX: Load default config to get map dimensions ---
    cfg = get_default_config()
    
    num_workers = os.cpu_count()
    print(f"Starting parallel normalization with {num_workers} workers...")
    episode_groups = df.groupby('episode_id')
    
    chunksize = max(1, len(episode_ids) // (num_workers * 4))
    print(f"Using chunksize: {chunksize}")
    
    processed_episodes_np = []
    # --- FIX: Use the initializer to pass config to all workers ---
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(cfg,)) as pool:
        with tqdm(total=len(episode_ids), desc="Normalizing Episodes") as pbar:
            for result in pool.imap_unordered(normalize_episode, episode_groups, chunksize=chunksize):
                processed_episodes_np.append(result)
                pbar.update()

    # --- Convert BOTH 'demands' and 'condition' to Tensors in the main process ---
    print("\nConverting NumPy arrays to PyTorch tensors...")
    processed_episodes = [
        {
            'id': ep['id'],
            'condition': torch.from_numpy(ep['condition']),
            'demands': torch.from_numpy(ep['demands'])
        }
        for ep in tqdm(processed_episodes_np, desc="Converting to Tensors")
    ]

    # Split data into train, val, test sets
    np.random.shuffle(processed_episodes)
    train_end = int(len(processed_episodes) * 0.8)
    val_end = int(len(processed_episodes) * 0.9)
    
    train_set = processed_episodes[:train_end]
    val_set = processed_episodes[train_end:val_end]
    test_set = processed_episodes[val_end:]

    print(f"\nDataset split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    torch.save((train_set, val_set, test_set), output_file)
    print(f"Saved normalized data to cache: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize DVRP demand dataset.")
    parser.add_argument(
        '--input', 
        type=str, 
        default='data/ruled_generator/demand_dataset_extended_diversity.csv',
        help='Path to the raw input CSV file.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/ruled_generator/normalized_dataset_extended_diversity.pt',
        help='Path to save the normalized output .pt file.'
    )
    args = parser.parse_args()
    
    process_data_parallel(args.input, args.output)