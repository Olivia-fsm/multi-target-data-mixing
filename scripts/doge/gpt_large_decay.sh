#!/bin/bash
cd /scratch/homes/sfan/multi_doge
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
pip install --upgrade wandb
# python src/run.py --config_json /scratch/homes/sfan/multi_doge/config/doge/gpt_base.json --wandb_run DOGE-6tasks-125M
# python src/run.py --config_json /scratch/homes/sfan/multi_doge/config/doge/gpt_base_dw50.json --wandb_run DOGE-6tasks-125M
# torchrun --nproc-per-node 4 src/run.py --config_json /scratch/homes/sfan/multi_doge/config/doge/gpt_large_dw100.json --wandb_run DOGE-6tasks-684M
torchrun --nproc_per_node 4 src/run.py --config_json /scratch/homes/sfan/multi_doge/config/doge/gpt_large_dw100_decay.json --wandb_run DOGE-8tasks-684M
# python src/run.py --config_json /scratch/homes/sfan/multi_doge/config/doge/gpt_large_dw100.json --wandb_run DOGE-6tasks-684M
