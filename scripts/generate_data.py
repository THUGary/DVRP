"""
This scirpt use the RuleBasedGenerator to generate a large dataset of DVRP episodes
with varied parameters, saving the results to a CSV file for later use in training
the neural network-based demand generator.
"""

import os
import csv
import random
import itertools
from typing import Dict, Any, Iterator, List, Tuple
import multiprocessing
from tqdm import tqdm
import sys

# Ensure the project root is in sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from agent.generator.rule_generator import RuleBasedGenerator
from configs import get_default_config, get_param_combinations



# --- Configuration for Data Generation ---

# Directory to save the generated dataset
DATASET_DIR = "/data/ruled_generator"
DATASET_NAME = "demand_dataset_extended_diversity.csv"
# Number of episodes to generate for EACH unique parameter combination
EPISODES_PER_COMBINATION = 100
# Maximum simulation time for each episode
MAX_TIME = 100



def generate_episode_data(args: Tuple[int, Dict[str, Any]]) -> List[List[Any]]:
    """
    Worker function to generate all data for a single episode.
    This function is designed to be run in a separate process.
    """
    episode_id, params = args
    episode_rows = []
    
    cfg = get_default_config()
    cfg.generator_params.update(params)
    cfg.generator_params["max_time"] = MAX_TIME
    
    generator = RuleBasedGenerator(
        width=cfg.width, height=cfg.height, **cfg.generator_params
    )
    
    # Use a random seed for each episode to ensure variety
    seed = random.randint(0, 2**32 - 1)
    generator.reset(seed=seed)

    for t in range(MAX_TIME):
        new_demands = generator.sample(t)
        if not new_demands:
            continue

        for demand in new_demands:
            row = [
                episode_id, t, demand.x, demand.y, demand.c, demand.end_t,
                params["total_demand"], params["num_centers"], params["distribution"],
                params["neighborhood_size"], params["max_c"], params["min_lifetime"],
                params["max_lifetime"], MAX_TIME
            ]
            episode_rows.append(row)
            
    return episode_rows


def generate_data_parallel():
    """
    Main function to generate the dataset in parallel using multiple CPU cores.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created dataset directory: {DATASET_DIR}")

    output_csv_path = os.path.join(DATASET_DIR, DATASET_NAME)
    
    param_combinations = list(get_param_combinations())
    
    # Create a list of all tasks (one for each episode)
    tasks = []
    episode_id_counter = 0
    for params in param_combinations:
        for _ in range(EPISODES_PER_COMBINATION):
            tasks.append((episode_id_counter, params))
            episode_id_counter += 1

    total_episodes = len(tasks)
    print(f"Found {len(param_combinations)} unique parameter combinations.")
    print(f"Total episodes to generate: {total_episodes}")
    
    # Use all available CPU cores
    num_workers = os.cpu_count()
    print(f"Starting data generation with {num_workers} parallel workers...")
    print(f"Saving data to: {output_csv_path}")

    csv_headers = [
        "episode_id", "time_step", "demand_x", "demand_y", "demand_c", "demand_end_t",
        "param_total_demand", "param_num_centers", "param_distribution",
        "param_neighborhood_size", "param_max_c", "param_min_lifetime",
        "param_max_lifetime", "param_max_time"
    ]

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for efficiency and tqdm for a progress bar
            results = pool.imap_unordered(generate_episode_data, tasks)
            
            for episode_rows in tqdm(results, total=total_episodes, desc="Generating Episodes"):
                if episode_rows:
                    writer.writerows(episode_rows)

    print(f"\nDataset generation complete. Data saved in {output_csv_path}")


if __name__ == "__main__":
    # It's good practice to rename the old main function and call the new parallel one
    generate_data_parallel()


