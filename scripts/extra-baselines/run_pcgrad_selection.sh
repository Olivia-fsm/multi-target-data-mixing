#!/bin/bash

CONDA_ENV=<>
GPUS_PER_NODE=2
PROJECT_DIR=<>
SRC_DIR="${PROJECT_DIR}/src"
CONFIG_DIR="${PROJECT_DIR}/config/extra-baselines/pcgrad"
WANDB_PROJ="multi-target-reweight"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Install requirements
pip install -r ${PROJECT_DIR}/requirements.txt
pip install --upgrade wandb

# Function to check if an item is in an array
contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i<$#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            return 0
        fi
    }
    return 1
}

# Check if arguments were provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide at least one configuration number (1-9)"
    echo "Usage: $0 [1-9] [1-9] ..."
    echo "Example: $0 1 3 7   # This will run T1, T3, and T7 configurations"
    exit 1
fi

# Validate all arguments are between 1 and 9
for arg in "$@"; do
    if ! [[ "$arg" =~ ^[1-9]$ ]]; then
        echo "Error: Arguments must be numbers between 1 and 9"
        exit 1
    fi
done

for config_num in "$@"; do
    config_file="T${config_num}-pcgrad.json"
    run_name="PCGRAD-T${config_num}"
    
    echo "Running configuration: $config_file with run name: $run_name"
    
    if [ ! -f "$CONFIG_DIR/$config_file" ]; then
        echo "Warning: Configuration file $CONFIG_DIR/$config_file does not exist. Skipping."
        continue
    fi
    
    # Run the command using torchrun
    torchrun --nproc-per-node $GPUS_PER_NODE $SRC_DIR/run_pcgrad.py \
        --config_json $CONFIG_DIR/$config_file \
        --wandb_run $run_name \
        --wandb_proj $WANDB_PROJ
    
    echo "Completed run for T${config_num}"
    sleep 2
done

echo "All requested configurations have been processed."
