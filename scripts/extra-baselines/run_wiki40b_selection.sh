#!/bin/bash

# Script to run T1-T2 configurations based on command-line arguments
# Usage: ./script.sh -m method [configuration numbers]
# Example: ./script.sh -m crisp 1 2   # This will run T1-crisp, and T2-crisp configurations
# Supported methods: crisp, regmix, pcgrad

CONDA_ENV="mtl"  
GPUS_PER_NODE=2
PROJECT_DIR="<>"
SRC_DIR="${PROJECT_DIR}/src"
CONFIG_DIR="${PROJECT_DIR}/config/ablations-wiki40b"
WANDB_PROJ="multi-target-reweight"

export WANDB_API_KEY=<>

# Default method
METHOD="pcgrad"

# Parse command line arguments
while getopts "m:" opt; do
  case ${opt} in
    m )
      METHOD=$OPTARG
      if [[ "$METHOD" != "crisp" && "$METHOD" != "regmix" && "$METHOD" != "pcgrad" ]]; then
        echo "Error: Method must be one of: crisp, regmix, pcgrad"
        exit 1
      fi
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

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

# Check if configuration numbers were provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide at least one configuration number (1-12)"
    echo "Usage: $0 -m method [1-2] [1-2] ..."
    echo "Example: $0 -m crisp 1 3 7   # This will run T1-crisp, T3-crisp, and T7-crisp configurations"
    echo "Supported methods: crisp, regmix, pcgrad"
    exit 1
fi

# Validate all arguments are between 1 and 2
for arg in "$@"; do
    if ! [[ "$arg" =~ ^[0-2]+$ ]]; then
        echo "Error: Arguments must be numbers between 1 and 2"
        exit 1
    fi
done

# Determine which script to run based on the method
if [[ "$METHOD" == "pcgrad" ]]; then
    RUN_SCRIPT="run_pcgrad.py"
else
    RUN_SCRIPT="run.py"
fi

for config_num in "$@"; do
    # Create the configuration filename and run name
    config_file="T${config_num}-${METHOD}.json"
    run_name="WIKI40b-${METHOD^^}-T${config_num}"
    
    echo "Running configuration: $config_file with run name: $run_name"
    
    # Check if the config file exists
    if [ ! -f "$CONFIG_DIR/$config_file" ]; then
        echo "Warning: Configuration file $CONFIG_DIR/$config_file does not exist. Skipping."
        continue
    fi
    
    # Run the command using torchrun
    torchrun --nproc-per-node $GPUS_PER_NODE $SRC_DIR/$RUN_SCRIPT \
        --config_json $CONFIG_DIR/$config_file \
        --wandb_run $run_name \
        --wandb_proj $WANDB_PROJ
    
    # Small delay between job submissions
    echo "Completed run for $METHOD T${config_num}"
    sleep 2
done

echo "All requested configurations have been processed."
