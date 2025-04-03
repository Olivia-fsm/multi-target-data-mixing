#!/bin/bash
cd /scratch/homes/sfan/multi_doge # replace to your directory
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
pip install --upgrade wandb

torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-BASE
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T1-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T1
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T2-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T2
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T3-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T3
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T4-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T4
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T5-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T5
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T6-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T6
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T7-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T7
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T8-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T8
torchrun --nproc-per-node 2 src/run.py --config_json config/ablations-doge/T9-gpt_base.json --wandb_proj multi-target-reweight --wandb_run ABLATION-DOGE-T9
