#!/bin/bash
cd /scratch/homes/sfan/multi_doge # replace to your directory
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
pip install --upgrade wandb

torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T1-gpt_base.json --wandb_run ABLATION-T1
torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T2-gpt_base.json --wandb_run ABLATION-T2
torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T3-gpt_base.json --wandb_run ABLATION-T3
torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T4-gpt_base.json --wandb_run ABLATION-T4
torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T5-gpt_base.json --wandb_run ABLATION-T5
torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T6-gpt_base.json --wandb_run ABLATION-T6
torchrun --nproc-per-node 4 src/run.py --config_json config/ablations/T7-gpt_base.json --wandb_run ABLATION-T7
