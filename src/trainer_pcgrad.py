import re
import os
import sys
import time
import shutil
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import random
import tempfile
import time
import warnings
from collections.abc import Mapping
import wandb
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union, TypedDict
import torch
import torch.distributed as dist
from torch.autograd import grad
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from datasets import IterableDataset

import transformers
from transformers import Trainer
from transformers.utils import ExplicitEnum, is_torch_tpu_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.optimization import get_scheduler
from transformers.utils import logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import (
        has_length,
        denumpify_detensorize,
        EvalLoopOutput,
        enable_full_determinism,
        set_seed,
        get_last_checkpoint,
        PREFIX_CHECKPOINT_DIR
)
from transformers.trainer import TRAINER_STATE_NAME
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers import (
    TrainingArguments, 
    MODEL_FOR_CAUSAL_LM_MAPPING,
    CONFIG_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state_for_device,
    # smp_forward_backward, 
    # smp_forward_only, 
    # smp_gather, 
    # smp_nested_concat
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from data import DataTrainingArguments, get_data_collator, get_train_eval_datasets, interleave_datasets
from accelerate import Accelerator, skip_first_batches
from accelerate import __version__ as accelerate_version
from accelerate.state import AcceleratorState
from accelerate.utils import (
    AutocastKwargs,
    DistributedDataParallelKwargs,
    DistributedType,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)

import sys
# sys.path.append("/scratch/homes/sfan/multi_doge/src")
from models import CausalLMOutputWithDomainIDs, ModelArguments, get_model_from_config, GPTForReweight
from schedulers import get_scheduler_extended
from ema import EMATracker


@dataclass
class FullTrainingArguments(TrainingArguments):
    lr_end: float = field(
            default=1e-4,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    reweight_train: str = field(
        default="doge", metadata={"help": "Reweighting training domains. [doge; doge_ema; doremi; None: uniform.]"}
    )
    reweight_tgt: str = field(
        default="pc_grad", metadata={"help": "Reweighting target domains. [map; map_gap; map_ema; xx_grad: manipulate graduents; None: fixed.]"}
    )
    reweight_train_iters: int = field(
        default=10, metadata={"help": "Frequency of training domains reweighting."}
    )
    reweight_tgt_iters: int = field(
        default=100, metadata={"help": "Frequency of target reweighting."}
    )
    mu_train: float = field(
        default=0.001,
        metadata={"help": "Hyperparam for Bregman Divergence on training domain weights (alpha)."}
    )
    mu_tgt: float = field(
        default=0.0001,
        metadata={"help": "Hyperparam for Bregman Divergence on target task weights (z)."}
    )
    reweight_batch_size: int = field(
        default=16,
        metadata={"help": "Batch Size for reweighting."}
    )
    reweight_mini_batch_size: int = field(
        default=8,
        metadata={"help": "Batch Size for reweighting."}
    )
    reweight_eps: float = field(
        default=0.0,
        metadata={"help": "Smoothing on domain weights."}
    )
    lr_scheduler_name: str = field(
        default='linear_warmup_cosine', metadata={"help": "Custom LR scheduler name (linear_warmup_exponential, linear_warmup_cosine, linear_warmup_decay)"}
    )
    ref_model: str = field(
        default=None, metadata={"help": "path to pretrained reference model."}
    )   # use for DoReMi or golden DRO
    dw_max: float = field(
        default=5.0,
        metadata={"help": "Score clip upper bound (*lr_t)."}
    )
    dw_min: float = field(
        default=0.00,
        metadata={"help": "Score clip lower bound (*lr_t)."}
    )
    ema_beta: float = field(
        default=0.9,
        metadata={"help": "beta for ema update."}
    )
    compute_pertoken_losses: bool = field(
        default=True, metadata={"help": "Compute all domain losses at once."}
    )
    ddp_backend: str = field(
        default=None, metadata={"help": "DDP Backend."}
    )
    include_num_input_tokens_seen: bool = field(
        default=True, metadata={"help": "Log num. training tokens."}
    )


### Trainer ###

from trainer import MAPTrainer
import logging
logger = logging.getLogger(__name__)


class MAPTrainerExtended(MAPTrainer):
    """
    Extension of MAPTrainer that:
     *  implements PCGrad for handling conflicting gradients between target domains during domain reweighting.
    

    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add PCGrad specific options
        self.pcgrad_epsilon = 1e-8  # Small value to avoid division by zero
        
        # Parse the reweight_tgt parameter to determine if PCGrad should be used
        self.use_pcgrad = "pc_grad" in self.reweight_tgt.lower()
        
        if self.use_pcgrad:
            logger.info(f"Using PCGrad for target gradient calculation with method: {self.reweight_tgt}")
            print(f"Using PCGrad for target gradient calculation with method: {self.reweight_tgt}")

    
        
    def apply_pcgrad(self, individual_tgt_inputs, max_domains=10):
        """
        Computes a consensus gradient using the Projection-based Conflicting Gradients (PCGrad) algorithm 
        for multi-domain optimization problems.
        
        This implementation follows the original PCGrad paper (Yu et al., 2020) and resolves conflicts
        between gradients from different target domains by projecting conflicting gradients onto the
        normal plane of other gradients. The resulting consensus gradient helps prevent negative transfer
        between domains during training.
        
        For computational efficiency, the function limits processing to a maximum number of randomly sampled
        target domains when more domains are available.
        
        Args:
            individual_tgt_inputs: Batch of inputs from target domains structured for domain-specific
                                gradient computation
            max_domains: Maximum number of target domains to sample 
        
        Returns:
            consensus_gradient: List of parameter-wise gradient tensors representing the 
                                consensus direction after resolving conflicts between domains
        """
        batch_size = len(individual_tgt_inputs[0]['input_ids'])
        # print(batch_size)
        # Sample a maximum number of target domains if there are more available
        max_domains = min(max_domains, len(self.tgt_ids))
        sampled_indices = random.sample(range(len(self.tgt_ids)), max_domains)
        projection_order = sampled_indices
        print("projection order:", projection_order)
        # First, calculate all domain gradients and store them
        target_gradients = []
        target_gradients_norms_sq = []
        for idx in sampled_indices:
                domain_grad = self.get_domain_grad(individual_tgt_inputs, domain_id=idx, 
                                                get_log_grad=False, return_loss=False)
                target_gradients.append(domain_grad)
                target_gradients_norms_sq.append(domain_grad.norm().square() + self.pcgrad_epsilon)
       
        
        consensus_gradient = torch.zeros_like(target_gradients[0])
        for i_idx in range(max_domains):
            current_grad = target_gradients[i_idx]
            for j_idx in range(max_domains):
                if i_idx != j_idx:
                    dot_product = torch.dot(current_grad, target_gradients[j_idx])
                    if dot_product < 0:
                        # Project grad_i onto the normal plane of grad_j
                        current_grad -= (dot_product / target_gradients_norms_sq[j_idx]) * target_gradients[j_idx]

            consensus_gradient += current_grad
        # Normalize the consensus gradient
        consensus_gradient /= len(sampled_indices)

        # clean up
        del target_gradients
        del target_gradients_norms_sq

        return consensus_gradient

    def log_gpu_memory(self, tag=""):
        """
        Simple function to log current GPU memory usage to wandb
        
        Args:
            tag: String to identify the logging point
        
        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available() or (self.args.local_rank != 0 and self.args.local_rank != -1):
            return {}
        
        memory_stats = {}
        # Log memory for each GPU
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # Convert to GB
            
            memory_stats[f"gpu{i}_mem_{tag}_allocated_gb"] = allocated
            memory_stats[f"gpu{i}_mem_{tag}_reserved_gb"] = reserved
        
        # Log to wandb if this is the main process
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            wandb.log(memory_stats, commit=False)
        
        return memory_stats

    # reweight train domains and target tasks
    def domain_reweight_step(self, 
                             individual_train_loader, 
                             tgt_loader,
                             lr_value,):
        wandb_log_dict = {}
        # tgt_grad calculation (depending on the reweight_tgt method) ###################
        if self.reweight_tgt == "pc_grad":
            individual_tgt_loader = tgt_loader
            individual_tgt_inputs = individual_tgt_loader.__next__()
            tgt_grad = self.apply_pcgrad(individual_tgt_inputs)
            
        elif  self.reweight_tgt == "map_single":   
            print("you shouldn't be here 😛")
            # Sample a single target domain based on tgt_dw probabilities
            sampled_domain_idx = torch.multinomial(self.tgt_dw, 1).item()
            sampled_domain_id = self.tgt_ids[sampled_domain_idx]
            domain_name = self.idx2domain[sampled_domain_id]
            
            # Get the data from the sampled domain
            individual_tgt_inputs = tgt_loader.__next__()
            
            # For logging
            print(f"[{self.state.global_step}] Sampled target domain: {domain_name} (index: {sampled_domain_idx})")

            # Get target gradient from the sampled domain
            get_tgt_log_grad = "map" in self.reweight_tgt
            tgt_grad = self.get_domain_grad(individual_tgt_inputs, 
                                            domain_id=sampled_domain_idx, 
                                            get_log_grad=get_tgt_log_grad)
        else: 
            mix_tgt_iter = tgt_loader
            num_batches = self.reweight_batch_size // self.reweight_mini_batch_size
            get_tgt_log_grad = "map" in self.reweight_tgt
            tgt_inputs, _ = self.get_batch_samples(mix_tgt_iter, num_batches)

            # for logging ### 
            sampled_domain_ids = []
            for batch in tgt_inputs:  # or train_inputs
                if "domain_ids" in batch:
                    sampled_domain_ids += [int(i) for i in batch["domain_ids"].flatten().tolist()]
            unique_domains = sorted(set(sampled_domain_ids))
            domain_names = [self.idx2domain[i] for i in unique_domains]
            print(f"[{self.state.global_step}] Mixed batch includes target domains: {domain_names} [total batches used : {len(tgt_inputs)}]")
            ### end logging ###

            tgt_grad = None
            for tgt_input in tgt_inputs:
                if tgt_grad is not None:
                    tgt_grad += self.get_domain_grad(tgt_input, domain_id=None, get_log_grad=get_tgt_log_grad)
                else:
                    tgt_grad = self.get_domain_grad(tgt_input, domain_id=None, get_log_grad=get_tgt_log_grad)
        
        ####### end tgt_grad calculation ########################

        individual_train_inputs = individual_train_loader.__next__()
        log_train_dw = torch.zeros_like(self.train_dw)
        domain_losses = []
        for idx, domain_id in enumerate(self.train_ids):
            domain_name = self.idx2domain[domain_id]
            domain_loss, domain_grad = self.get_domain_grad(individual_train_inputs, domain_id=idx, get_log_grad=False, return_loss=True)
            domain_losses.append(domain_loss)
            wandb_log_dict[f'grad_norm/{domain_name}'] = domain_grad.norm().item()
            self.train_dw_update_counter[domain_id] += len(individual_train_inputs[idx])
            wandb_log_dict[f'train_reweight_count/{domain_name}'] = self.train_dw_update_counter[domain_id]
            log_train_dw[idx] = torch.log(self.train_dw[idx]) + lr_value * (domain_grad @ tgt_grad) / self.mu_train
        del domain_grad
        del tgt_grad
        train_dw = torch.nn.functional.softmax(log_train_dw, dim=-1)
        self.train_dw_update_steps += 1
        self.train_dw = train_dw
        self.avg_train_dw += train_dw
        for idx, domain_id in enumerate(self.train_ids):
            domain_name = self.idx2domain[domain_id]
            wandb_log_dict[f'avg_train_dw/{domain_name}'] = self.avg_train_dw[idx].item() / self.train_dw_update_steps
            wandb_log_dict[f'train_dw/{domain_name}'] = self.train_dw[idx].item()
            ema_domain_loss = self.train_tracker_dict[domain_name].update(step=self.state.global_step,
                                                                          loss=domain_losses[idx],
                                                                          score=self.train_dw[idx].item())
            wandb_log_dict[f'reweight_loss/{domain_name}'] = domain_losses[idx],
            wandb_log_dict[f'ema_reweight_loss/{domain_name}'] = ema_domain_loss
            self.train_tracker_dict[domain_name].save()
        
        self.write_weights(last_train_dw=self.train_dw, 
                           avg_train_dw=self.avg_train_dw/self.train_dw_update_steps,
                           last_tgt_dw=self.tgt_dw,
                           avg_tgt_dw=self.avg_tgt_dw/self.tgt_dw_update_steps,
                           train_dw_update_steps=self.train_dw_update_steps,
                           tgt_dw_update_steps=self.tgt_dw_update_steps)
        
        self.train_loader.update_weights(new_weights=self.train_dw)
        logger.info("-> update train_dw: ", self.train_dw)
        print("-> update train_dw: ", self.train_dw)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            wandb.log(wandb_log_dict, commit=False)
    
  
