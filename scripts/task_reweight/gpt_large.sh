#!/bin/bash
cd /scratch/homes/sfan/multi_doge
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
pip install --upgrade wandb
torchrun --nproc-per-node 4 src/run.py --config_json /scratch/homes/sfan/multi_doge/config/task_reweight/gpt_large_dw100.json --wandb_run MAP-6tasks-684M