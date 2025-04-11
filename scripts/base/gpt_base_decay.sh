#!/bin/bash
cd /scratch/homes/sfan/multi_doge
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
pip install --upgrade wandb
# python src/run.py --config_json /scratch/homes/sfan/multi_doge/config/base/gpt_base.json --wandb_run BASE-6tasks-125M
# torchrun --nproc-per-node 2 src/run.py --config_json /scratch/homes/sfan/multi_doge/config/base/gpt_large_decay.json --wandb_run BASE-9tasks-684M-new
torchrun --nproc-per-node 4 src/run.py --config_json /scratch/homes/sfan/multi_doge/config/base/gpt_large_decay.json --wandb_run BASE-new_tasks
