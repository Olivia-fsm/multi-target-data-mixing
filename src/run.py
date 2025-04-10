from trainer import *
import warnings
warnings.filterwarnings("ignore")
import logging
from pathlib import Path
import os
import sys
import json
import numpy as np
import argparse
import datasets
import torch
import pickle

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import datasets
from datasets import concatenate_datasets
from data import DataTrainingArguments, DomainConfigArguments, get_data_collator, get_train_eval_datasets
from models import CausalLMOutputWithDomainIDs, ModelArguments, get_model_from_config, GPTForReweight
from trainer import FullTrainingArguments

args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--config_json', default='/mloraw1/sfan/doge/config/doge_82.json', type=str)
args_parser.add_argument('--wandb_proj', default='multi-target-reweight_684M', type=str)
args_parser.add_argument('--wandb_run', default=None, type=str)

def get_run_name(base_name="", training_args={}):
    train_dw_iter = training_args.reweight_train_iters
    base_name += f"-dw[{train_dw_iter}]"
    tgt_dw_iter = training_args.reweight_tgt_iters
    base_name += f"-tw[{tgt_dw_iter}]"
    scheduler_name = training_args.lr_scheduler_name
    scheduler_name = scheduler_name.split("_")[-1]
    base_name += f"-scheduler[{scheduler_name}]"
    return base_name

# Custom progress bar callback that only shows on main process
class TqdmCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # Store whether we're on the main process
        self.is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
        self.running_pbar = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        if self.is_main:
            total_steps = state.max_steps if state.max_steps > 0 else state.num_train_epochs * state.steps_per_epoch
            self.running_pbar = tqdm(
                total=total_steps,
                desc="Training",
                # Disable if not in interactive terminal
                disable=not sys.stderr.isatty()
            )
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.is_main and self.running_pbar is not None:
            self.running_pbar.update(1)
            # Update description with loss
            if state.log_history:
                latest_loss = state.log_history[-1].get("loss", None)
                if latest_loss is not None:
                    self.running_pbar.set_description(
                        f"Training (loss: {latest_loss:.4f})"
                    )
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.is_main and self.running_pbar is not None:
            self.running_pbar.close()
                
                
def main():
    args = args_parser.parse_args()
    dist.init_process_group('nccl')
    local_rank = torch.distributed.get_rank()
    local_world_size = os.environ["LOCAL_WORLD_SIZE"]                                                                                                                           
    os.environ["WANDB_PROJECT"] = args.wandb_proj # name your W&B project 
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ["WANDB_START_METHOD"] = "thread"
    config_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FullTrainingArguments))
    if args.config_json is not None:
        model_args, data_args, training_args = config_parser.parse_json_file(json_file=args.config_json)
    else:
        model_args, data_args, training_args = config_parser.parse_args_into_dataclasses()
    
    if args.wandb_run is None:
        wandb_run_name = training_args.run_name
    else:
        wandb_run_name = args.wandb_run
    wandb_run_name = get_run_name(base_name=wandb_run_name, training_args=training_args)
    # Initialize wandb - do this only on the main process
    # if os.environ.get("LOCAL_RANK", "0") == "0":
    #     wandb.init(
    #         project=args.wandb_proj,
    #         name=wandb_run_name,
    #         # Other wandb configurations as needed
    #     )
    
    # enable NCCL 
    if local_rank != -1:
        ddp_backend = "nccl"
    else:
        ddp_backend = None

    training_args.local_rank = local_rank
    training_args.ddp_backend = ddp_backend
    print(f'{local_rank = }| { local_world_size = }')
    print("training local_rank: ", training_args.local_rank)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    individual_train_ds, individual_tgt_ds, individual_val_ds, domain_config, tokenizer = get_train_eval_datasets(data_config=data_args,
                                                            verbose=True,)
    data_collator=get_data_collator(tokenizer, do_padding=data_args.do_padding, max_length=data_args.max_token_length)
    
    # save for future
    ref_model = None
    
    ## Start Training ##
    # Detecting last checkpoint.
    reweight_model, reweight_config = get_model_from_config(model_args, reweight=True)
    print("model parameters: ", reweight_model.num_parameters())
    print("Num. GPU used: ", training_args.n_gpu)
    print("Gradient accumulate steps: ", training_args.gradient_accumulation_steps)
    last_checkpoint = None
    num_skip_examples = 0
    output_dir = os.path.join(training_args.output_dir, wandb_run_name)
    if os.path.isdir(output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            state = TrainerState.load_from_json(str(Path(last_checkpoint) / TRAINER_STATE_NAME))
            global_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
            num_skip_examples = state.global_step * global_batch_size
            print(f"Skipping {num_skip_examples} examples")
    else:
        os.makedirs(output_dir, exist_ok=True)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    torch.cuda.empty_cache()
    # Initialize our Trainer
    trainer = MAPTrainer(
        model=reweight_model,
        args=training_args,
        domain_args=domain_config,
        train_dataset_ls=individual_train_ds,
        tgt_dataset_ls=individual_tgt_ds,
        val_dataset_ls=individual_val_ds,
        eval_dataset=individual_val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        wandb_run_name=wandb_run_name,
        output_dir=output_dir,
        ref_model=ref_model,
        # callbacks=[TqdmCallback],
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     if training_args.eval_all_checkpoints:
    #         checkpoint_dir_list = trainer.get_all_checkpoints(training_args.output_dir)
    #     else:
    #         checkpoint_dir_list = [get_last_checkpoint(training_args.output_dir)]

    #     for checkpoint_dir in checkpoint_dir_list:
    #         trainer.load_checkpoint(checkpoint_dir)
    #         state = TrainerState.load_from_json(str(Path(checkpoint_dir) / TRAINER_STATE_NAME))
    #         trainer.state.global_step = state.global_step
    # if os.environ.get("LOCAL_RANK", "0") == "0":
    #     wandb.finish()
    logger.info('MAP launched! ❤️‍🔥❤️‍🔥')
        
if __name__ == "__main__":
    main()