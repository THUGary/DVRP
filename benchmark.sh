#!/usr/bin/env bash
# Simple wrapper to run the static benchmark with the provided checkpoint
# Usage: ./benchmark.sh

# Choose condition, condtion=0 for a single instance test,
# condtion=2 for a single instance test with dynamic adapter,
# condtion=1 for full benchmark (on solomon)
condition=3

# Another arguments
FPS=${FPS:-10}
Planner_Choice=${Planner_Choice:-static} # static| dcp | greedy | fri | rbso | dynamic (model) |
Static_Ckpt=${Static_Ckpt:-./checkpoints/static_vrp_v2/best_n80.pt}
Adapter_Ckpt=${Adapter_Ckpt:-./checkpoints/dynamic_adapter_v2_new/best_adapter_rl.pt}

if [ $condition -eq 0 ]; then
    echo "Running single instance test..."
    python benchmark.py \
        --render --fps ${FPS} \
        --service-time \
        --static-ckpt ${Static_Ckpt} \
        --static-demands \
    
    exit 0
elif [ $condition -eq 3 ]; then
    echo "Running single instance test with prompt planner..."
    python benchmark.py \
        --render --fps ${FPS} \
        --service-time \
        --prompt-planner \
        --static-demands \
    
    exit 0
elif [ $condition -eq 2 ]; then
    echo "Running single instance test with dynamic adapter..."
    python benchmark.py \
        --render --fps ${FPS} \
        --service-time \
        --adapter-ckpt ${Adapter_Ckpt} \
    
    exit 0
elif [ $condition -eq 1 ]; then
    echo "Running full benchmark on Solomon instances..."
    # Add commands for full benchmark here
    python benchmark.py \
        --static-demands \
        --service-time \
        --static-ckpt ${Static_Ckpt} \
        --least-vehs \
        --test-all
        # least-vehs use the number of vehicles in the best solution as agent number
    exit 0
fi
    