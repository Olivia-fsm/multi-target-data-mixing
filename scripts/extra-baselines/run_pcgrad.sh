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

export WANDB_API_KEY=<>

# List of configuration files and their corresponding run names
CONFIG_FILES=(
  "T8-pcgrad.json:PCGRAD-T8"
)
for config_run in "${CONFIG_FILES[@]}"; do
  IFS=':' read -r config_file run_name <<< "$config_run"
  
  torchrun --nproc-per-node $GPUS_PER_NODE $SRC_DIR/run_pcgrad.py --config_json $CONFIG_DIR/$config_file --wandb_run $run_name --wandb_proj $WANDB_PROJ
  
  sleep 2
done
