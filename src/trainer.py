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
sys.path.append("/scratch/homes/sfan/multi_doge/src")
from models import CausalLMOutputWithDomainIDs, ModelArguments, get_model_from_config, GPTForReweight
from schedulers import get_scheduler_extended
from ema import EMATracker

@dataclass
class FullTrainingArguments(TrainingArguments):
    lr_end: float = field(
            default=1e-4,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    decay_rate: float = field(
            default=0.1,
            metadata={"help": "The decay ratio of WSD lr scheduler."},
    )
    reweight_train: str = field(
        default="doge", metadata={"help": "Reweighting training domains. [doge; doge_ema; doremi; None: uniform.]"}
    )
    reweight_tgt: str = field(
        default="map", metadata={"help": "Reweighting target domains. [map; map_gap; map_ema; map_star; xx_grad: manipulate graduents; None: fixed.]"}
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
import logging
logger = logging.getLogger(__name__)

class MAPTrainer(Trainer):
    def __init__(self, *args, domain_args, 
                 wandb_run_name="test_test_test", 
                 output_dir=None, grad_acc=None, ref_model=None, 
                 train_dataset_ls=None, tgt_dataset_ls=None, val_dataset_ls=None,
                 max_eval_steps=None,
                 **kwargs,):
        ''' args to init the original Trainer
          model: Union[PreTrainedModel, nn.Module] = None,
          args: TrainingArguments = None,
          domain_args: DomainConfigArguments = None,
          data_collator: Optional[DataCollator] = None,
          train_dataset: Optional[Dataset] = None,
          eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
          tokenizer: Optional[PreTrainedTokenizerBase] = None,
          model_init: Optional[Callable[[], PreTrainedModel]] = None,
          compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
          callbacks: Optional[List[TrainerCallback]] = None,
          optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
          preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        '''
        super().__init__(*args, **kwargs)
        
        self.domain_config = domain_args
        self.train_dataset_ls = train_dataset_ls
        self.tgt_dataset_ls = tgt_dataset_ls
        self.val_dataset_ls = val_dataset_ls
        self.eval_dataset = val_dataset_ls
        self.train_dw = self.domain_config.train_dw
        self.tgt_dw = self.domain_config.tgt_dw
        self.val_dw = self.domain_config.val_dw
        self.train_ids = self.domain_config.train_ids.tolist()
        self.tgt_ids = self.domain_config.tgt_ids.tolist()
        self.val_ids = self.domain_config.val_ids.tolist()

        self.idx2domain = self.domain_config.idx2domain
        self.domain2idx = self.domain_config.domain2idx
        self.domain_list = self.domain_config.domain_list
        # if self.domain_config
        
        self.train_domains = [self.idx2domain[d] for d in self.train_ids]
        self.tgt_domains = [self.idx2domain[d] for d in self.tgt_ids]
        self.eval_domains = [self.idx2domain[d] for d in self.val_ids]
        
        self.reweight_eps = self.args.reweight_eps
        self.reweight_batch_size = self.args.reweight_batch_size
        self.reweight_mini_batch_size = self.args.reweight_mini_batch_size
        self.reweight_train = self.args.reweight_train # str
        self.reweight_tgt = self.args.reweight_tgt # str
        self.reweight_train_iters = self.args.reweight_train_iters # int
        self.reweight_tgt_iters = self.args.reweight_tgt_iters # int
        self.ref_model = ref_model
        
        self.mu_train = self.args.mu_train
        self.mu_tgt = self.args.mu_tgt
        self.compute_pertoken_losses = self.args.compute_pertoken_losses
        if grad_acc is not None:
            self.args.gradient_accumulation_steps = grad_acc
            
        self.token_masks = []
        self.domain_ids = []
        self.num_grad_params = self.model.num_parameters()
        self.grad_acc_step = 0
        self.max_eval_steps = 1000 if max_eval_steps is None else max_eval_steps
        
        self.avg_train_dw = torch.zeros(len(self.train_ids), dtype=torch.float)
        self.avg_tgt_dw = torch.zeros(len(self.tgt_ids), dtype=torch.float)
        
        self.args.run_name = wandb_run_name
        if output_dir is not None:
            self.args.output_dir = output_dir 
        print(f'Training for {self.args.max_steps} Steps')
        print(f"-> Output dir: {self.args.output_dir}")
        
        self.ema_beta = self.args.ema_beta
        self.train_tracker_dict = {name:EMATracker(self.ema_beta, 
                                                   save_dir=self.args.output_dir,
                                                   name=name) for name in self.train_domains}
        self.tgt_tracker_dict = {name:EMATracker(self.ema_beta, 
                                                 save_dir=self.args.output_dir,
                                                 name=name) for name in self.tgt_domains}
        
        self.train_dw_update_steps = 0
        self.tgt_dw_update_steps = 0
        self.train_dw_update_counter = {i:0 for i in range(len(self.domain_list))}
        self.tgt_dw_update_counter = {i:0 for i in range(len(self.domain_list))}
        self.dw_save_path = os.path.join(self.args.output_dir, 'dw_config.pkl')
        self.reload_weights()
        
        if self.ref_model is not None:
            print("** Reference Model **")
            print(self.ref_model)
        print('==============================')
        print('Train Domains [index|name|weight]')
        for idx,i in enumerate(self.train_ids):
            print(f'{i}-{self.idx2domain[i]}|{self.train_dw[idx]}')
        print('\nTarget Tasks [index|name|weight]')
        for idx,i in enumerate(self.tgt_ids):
            print(f'{i}-{self.idx2domain[i]}|{self.tgt_dw[idx]}')
        print('==============================')
        
        self.get_individual_train_dataloader()
        self.get_individual_tgt_dataloader()
        self.get_train_dataloader()
        self.get_tgt_dataloader()
        
           
    def reload_weights(self):
        if os.path.exists(self.dw_save_path):
            with open(self.dw_save_path, 'rb') as trg:
                reload_dw_cfg = pickle.load(trg)
            self.train_dw = reload_dw_cfg['last_train_dw']
            self.tgt_dw = reload_dw_cfg['last_tgt_dw']
            self.train_dw_update_steps = reload_dw_cfg['train_dw_update_steps']
            self.tgt_dw_update_steps = reload_dw_cfg['tgt_dw_update_steps']
            self.avg_train_dw = reload_dw_cfg['avg_train_dw'] * self.train_dw_update_steps
            self.avg_tgt_dw = reload_dw_cfg['avg_tgt_dw'] * self.tgt_dw_update_steps
            
            print(f'Resume train/tgt domain weights from step [{self.train_dw_update_steps}|{self.tgt_dw_update_steps}]...')
            print('==============================')
            print('Last-step Train Domain Weights:', self.train_dw)
            print('Last-step Target (Task) Domain Weights:', self.tgt_dw)
            print('Average Train Domain Weights:', self.avg_train_dw / self.train_dw_update_steps)
            print('Average Target Task Weights:', self.avg_tgt_dw / self.tgt_dw_update_steps)
            print('==============================')
        
    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def write_weights(self, 
                      last_train_dw, 
                      avg_train_dw,
                      last_tgt_dw,
                      avg_tgt_dw,
                      train_dw_update_steps,
                      tgt_dw_update_steps,
                      save_path=None):
        if save_path is None:
            save_path = self.dw_save_path
        dw_config_dict = {k:v for k,v in self.domain_config.__dict__.items()}
        dw_config_dict['last_train_dw'] = last_train_dw
        dw_config_dict['avg_train_dw'] = avg_train_dw
        dw_config_dict['last_tgt_dw'] = last_tgt_dw
        dw_config_dict['avg_tgt_dw'] = avg_tgt_dw
        dw_config_dict['train_dw_update_steps'] = train_dw_update_steps
        dw_config_dict['tgt_dw_update_steps'] = tgt_dw_update_steps
        with open(save_path, 'wb') as trg:
            pickle.dump(dw_config_dict, trg)
        
    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        # if self.lr_scheduler is None:
        if self.args.lr_scheduler_name is not None:
            lr_scheduler_name = self.args.lr_scheduler_name
        else:
            lr_scheduler_name = self.args.lr_scheduler_type
        
        if lr_scheduler_name == "linear_decay":
            self.lr_scheduler = get_scheduler_extended(
                lr_scheduler_name,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=int(self.args.decay_rate*num_training_steps),
                lr_end=self.args.lr_end,
                decay_rate=self.args.decay_rate
            )
        else:
            self.lr_scheduler = get_scheduler_extended(
                lr_scheduler_name,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                lr_end=self.args.lr_end,
                decay_rate=self.args.decay_rate
            )
        self._created_lr_scheduler = True
        return self.lr_scheduler
    
    def compute_loss(self, model, inputs, return_outputs=True, return_pertoken_losses=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs['return_pertoken_losses'] = return_pertoken_losses
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def _get_train_sampler(self):
        return None
    
    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from data import interleave_dataloader, individual_dataloader
        import datasets
        
        train_dataset_ls = self.train_dataset_ls
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset_ls[0], datasets.Dataset):
            train_dataset_ls = [self._remove_unused_columns(ds, description="training") for ds in train_dataset_ls]
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset_ls[0], torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if self.train_dataset is None:
            logger.info(
                    "You are loading interleaving and individual datasets for `Domain Reweighting`"
                )
            if self.train_dataset_ls is None:
                raise ValueError("Trainer: reweighting requires a train_dataset_ls.")
            self.train_loader = interleave_dataloader(self.train_dataset_ls, self.train_dw, data_collator=data_collator, **dataloader_params)
            # self.train_loader = self.accelerator.prepare(self.train_loader)

        logger.info(
            "Setup mixture [train_loader]!💡"
        )
        return self.train_loader
    
    def get_tgt_dataloader(self):
        """
        Returns the target [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from data import interleave_dataloader, individual_dataloader
        import datasets
        
        if self.tgt_dataset_ls is not None:
            tgt_dataset_ls = self.tgt_dataset_ls
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(tgt_dataset_ls[0], datasets.Dataset):
                tgt_dataset_ls = [self._remove_unused_columns(ds, description="training") for ds in tgt_dataset_ls]
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(tgt_dataset_ls[0], torch.utils.data.IterableDataset):
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            self.tgt_loader = interleave_dataloader(self.tgt_dataset_ls, self.tgt_dw, data_collator=data_collator, **dataloader_params)            
            # self.tgt_loader = self.accelerator.prepare(self.tgt_loader)
        else:
            self.tgt_loader = None
        
        logger.info(
            "Setup mixture [tgt_loader]!📍"
        )
        return self.tgt_loader
    
    def get_eval_dataloader(self, val_dataset_ls=None):
        """
        Returns the target [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from data import interleave_dataloader, individual_dataloader
        import datasets
        
        if val_dataset_ls is None:
            val_dataset_ls = self.val_dataset_ls
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(val_dataset_ls[0], datasets.Dataset):
            val_dataset_ls = [self._remove_unused_columns(ds, description="validation") for ds in val_dataset_ls]
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="validation")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(val_dataset_ls[0], torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        self.val_loader = interleave_dataloader(self.val_dataset_ls, self.val_dw, data_collator=data_collator, **dataloader_params)            
        self.val_loader = self.accelerator.prepare(self.val_loader)
        
        logger.info(
            "Setup mixture [tgt_loader]!📍"
        )
        return self.val_loader

    def get_individual_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from data import interleave_dataloader, individual_dataloader
        import datasets
        
        train_dataset_ls = self.train_dataset_ls
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset_ls[0], datasets.Dataset):
            train_dataset_ls = [self._remove_unused_columns(ds, description="training") for ds in train_dataset_ls]
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self.reweight_mini_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset_ls[0], torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if self.train_dataset is None:
            logger.info(
                    "You are loading interleaving and individual datasets for `Domain Reweighting`"
                )
            if self.train_dataset_ls is None:
                raise ValueError("Trainer: reweighting requires a train_dataset_ls.")
            self.individual_train_loader = individual_dataloader(self.train_dataset_ls, 
                                                                 data_collator=data_collator, **dataloader_params)
            self.individual_train_loader = self.accelerator.prepare(self.individual_train_loader)

        logger.info(
            "Setup individual [train_loader]!💡"
        )
        return self.individual_train_loader
    
    def get_individual_tgt_dataloader(self):
        """
        Returns the target [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from data import interleave_dataloader, individual_dataloader
        import datasets
        if self.tgt_dataset_ls is not None:
            tgt_dataset_ls = self.tgt_dataset_ls
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(tgt_dataset_ls[0], datasets.Dataset):
                tgt_dataset_ls = [self._remove_unused_columns(ds, description="training") for ds in tgt_dataset_ls]
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                "batch_size": self.reweight_mini_batch_size,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(tgt_dataset_ls[0], torch.utils.data.IterableDataset):
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            self.individual_tgt_loader = individual_dataloader(self.tgt_dataset_ls, data_collator=data_collator, **dataloader_params)            
            self.individual_tgt_loader = self.accelerator.prepare(self.individual_tgt_loader)
        else:
            self.individual_tgt_loader = None
        
        logger.info(
            "Setup individual [tgt_loader]!📍"
        )
        return self.individual_tgt_loader
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False, return_pertoken_losses=False)
            # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
    
    def get_grad_backward(self, loss_value, flatten=True, return_dict=False):
        ''' Get model gradient with backward computation map. '''
        self.model.zero_grad()
        loss_value.backward()
        max_grad_norm = self.args.max_grad_norm
        # Gradient clipping
        if max_grad_norm is not None and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
            )
        # make sure all grads are detached #
        grad_vec_dict = {}
        for p_name, p in self.model.named_parameters():
            # flat_grad = p.grad.detach().flatten().cpu()
            p_grad = p.grad.detach()
            if flatten:
                p_grad = p_grad.flatten()
            grad_vec_dict[p_name] = p_grad
        if not return_dict:
            concat_grad = torch.concat([grad_vec_dict[p_name] for p_name, _ in self.model.named_parameters()])
            return concat_grad
        self.model.zero_grad()
        return grad_vec_dict
    
    def get_grad_forward(self, loss_value, flatten=True, return_dict=False):
        ''' Get model gradient with forward compute. '''
        # make sure all grads are detached #
        grad_vec_dict = {}
        for p_name, p in self.model.named_parameters():
            p_grad = torch.autograd.grad(loss_value, p, retain_graph=True)[0].detach()
            if flatten:
                p_grad = p_grad.flatten()
            grad_vec_dict[p_name] = p_grad
        if not return_dict:
            concat_grad = torch.concat([grad_vec_dict[p_name] for p_name, p in self.model.named_parameters()])
            return concat_grad
        self.model.zero_grad()
        return grad_vec_dict
    
    def get_domain_grad(self, individual_inputs, 
                        domain_id=None, 
                        get_log_grad=False,
                        return_loss=False):
        if domain_id is None:
            domain_inputs = individual_inputs
        else:
            domain_inputs = individual_inputs[domain_id]
        domain_inputs = self._prepare_inputs(domain_inputs)
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(self.model, domain_inputs, return_outputs=True, return_pertoken_losses=False)
        del domain_inputs
        loss_value = outputs.log_loss if get_log_grad else loss
        # compute gradient
        # domain_grad = self.get_grad_forward(loss_value, flatten=True, return_dict=False)
        domain_grad = self.get_grad_backward(loss_value, flatten=True, return_dict=False)
        
        if return_loss:
            return loss_value.detach().cpu().item(), domain_grad
        return domain_grad
    
    # reweight train domains and target tasks
    def domain_reweight_step(self, 
                             individual_train_loader, 
                             mix_tgt_iter,
                             lr_value,):
        wandb_log_dict = {}
        # TODO: implement doge with re-sampling
        # tgt_input = mix_tgt_loader.__next__()
        num_batches = self.reweight_batch_size // self.reweight_mini_batch_size
        # get_tgt_log_grad = (self.reweight_tgt == "map")
        get_tgt_log_grad = (self.reweight_tgt == "map") or (self.reweight_tgt == "map_ema")
        tgt_inputs, _ = self.get_batch_samples(mix_tgt_iter, num_batches)
        tgt_grad = None
        for tgt_input in tgt_inputs:
            if tgt_grad is not None:
                tgt_grad += self.get_domain_grad(tgt_input, domain_id=None, get_log_grad=get_tgt_log_grad)
            else:
                tgt_grad = self.get_domain_grad(tgt_input, domain_id=None, get_log_grad=get_tgt_log_grad)
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
        if (self.state.global_step+1) % self.args.save_steps == 0:
            self.write_weights(last_train_dw=self.train_dw, 
                                avg_train_dw=self.avg_train_dw/self.train_dw_update_steps,
                                last_tgt_dw=self.tgt_dw,
                                avg_tgt_dw=self.avg_tgt_dw/self.tgt_dw_update_steps,
                                train_dw_update_steps=self.train_dw_update_steps,
                                tgt_dw_update_steps=self.tgt_dw_update_steps,
                                save_path = os.path.join(f"{self.dw_save_path.split('.pkl')[0]}-{str(self.state.global_step)}.pkl"))
        
        self.train_loader.update_weights(new_weights=self.train_dw)
        logger.info("-> update train_dw: ", self.train_dw)
        print("-> update train_dw: ", self.train_dw)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            wandb.log(wandb_log_dict, commit=False)
    
    def tgt_reweight_step(self, 
                          individual_tgt_loader,
                          mix_train_iter,
                          lr_value):
        # TODO: implement doge with re-sampling
        wandb_log_dict = {}
        num_batches = self.reweight_batch_size // self.reweight_mini_batch_size
        get_tgt_log_grad = (self.reweight_tgt == "map")
        train_inputs, _ = self.get_batch_samples(mix_train_iter, num_batches)
        train_grad = None
        for train_input in train_inputs:
            if train_grad is not None:
                train_grad += self.get_domain_grad(train_input, domain_id=None, get_log_grad=False)
            else:
                train_grad = self.get_domain_grad(train_input, domain_id=None, get_log_grad=False)
        individual_tgt_inputs = individual_tgt_loader.__next__()
        log_tgt_dw = torch.zeros_like(self.tgt_dw)
        
        for idx, domain_id in enumerate(self.tgt_ids):
            domain_name = self.idx2domain[domain_id]
            domain_loss, domain_grad = self.get_domain_grad(individual_tgt_inputs, 
                                                            domain_id=idx, 
                                                            get_log_grad=get_tgt_log_grad, 
                                                            return_loss=True)
            wandb_log_dict[f'grad_norm/{domain_name}'] = domain_grad.norm().item()
            self.tgt_dw_update_counter[domain_id] += len(individual_tgt_inputs[idx])
            wandb_log_dict[f'tgt_reweight_count/{domain_name}'] = self.tgt_dw_update_counter[domain_id]
            
            ema_domain_loss = self.tgt_tracker_dict[domain_name].update(step=self.state.global_step,
                                                                        loss=domain_loss,
                                                                        score=None)
            wandb_log_dict[f'reweight_loss/{domain_name}'] = domain_loss
            wandb_log_dict[f'ema_reweight_loss/{domain_name}'] = ema_domain_loss
            if self.reweight_tgt == "map_ema":
                domain_grad = domain_grad / ema_domain_loss
            log_tgt_dw[idx] = torch.log(self.tgt_dw[idx]) - lr_value * (domain_grad @ train_grad) / self.mu_tgt
        del domain_grad
        del train_grad
        tgt_dw = torch.nn.functional.softmax(log_tgt_dw, dim=-1)
        self.tgt_dw_update_steps += 1
        self.tgt_dw = tgt_dw
        self.avg_tgt_dw += tgt_dw
        for idx, domain_id in enumerate(self.tgt_ids):
            domain_name = self.idx2domain[domain_id]
            wandb_log_dict[f'avg_tgt_dw/{domain_name}'] = self.avg_tgt_dw[idx].item() / self.tgt_dw_update_steps
            wandb_log_dict[f'tgt_dw/{domain_name}'] = self.tgt_dw[idx].item()
            self.tgt_tracker_dict[domain_name].update_score(step=self.state.global_step,
                                                            score=self.tgt_dw[idx].item())
            self.tgt_tracker_dict[domain_name].save()
        
        self.write_weights(last_train_dw=self.train_dw, 
                           avg_train_dw=self.avg_train_dw/self.train_dw_update_steps,
                           last_tgt_dw=self.tgt_dw,
                           avg_tgt_dw=self.avg_tgt_dw/self.tgt_dw_update_steps,
                           train_dw_update_steps=self.train_dw_update_steps,
                           tgt_dw_update_steps=self.tgt_dw_update_steps)
        if (self.state.global_step+1) % self.args.save_steps == 0:
            self.write_weights(last_train_dw=self.train_dw, 
                                avg_train_dw=self.avg_train_dw/self.train_dw_update_steps,
                                last_tgt_dw=self.tgt_dw,
                                avg_tgt_dw=self.avg_tgt_dw/self.tgt_dw_update_steps,
                                train_dw_update_steps=self.train_dw_update_steps,
                                tgt_dw_update_steps=self.tgt_dw_update_steps,
                                save_path = os.path.join(f"{self.dw_save_path.split('.pkl')[0]}-{str(self.state.global_step)}.pkl"))
        
        self.tgt_loader.update_weights(new_weights=self.tgt_dw)
        logger.info("-> update tgt_dw: ", self.tgt_dw)
        print("-> update tgt_dw: ", self.tgt_dw)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            wandb.log(wandb_log_dict, commit=False)
    
    #################################################################
    
    def load_checkpoint(self, resume_from_checkpoint=None):
        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(None)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint is None:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)

        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None and not is_sagemaker_mp_enabled() and self.args.deepspeed is None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, self.args.device)
            self.model_wrapped = self.model

    def get_all_checkpoints(self, folder):
        _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
        folder = Path(folder)
        checkpoints = [
            path
            for path in folder.iterdir()
            if _re_checkpoint.search(path.name) is not None and path.is_dir()
        ]
        checkpoints = list(sorted(checkpoints, key=lambda x: int(x.name.split('-')[1])))
        checkpoints = [str(path) for path in checkpoints]
        return checkpoints

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Computes per-domain log-perplexity, uniformly averaged log-perplexity, and worst-case log-perplexity
        """
        args = self.args

        if prediction_loss_only:
            # hack - don't do prediction loss only
            prediction_loss_only = None

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        if args.past_index >= 0:
            self._past = None

        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        losses = torch.zeros(len(self.eval_domains)).cuda()
        tokencounts = torch.zeros(len(self.eval_domains)).cuda()
        examplecounts = torch.zeros(len(self.eval_domains)).cuda()
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            domain_ids = inputs["domain_ids"].to(loss.device)

            if isinstance(logits, tuple):
                logits = logits[0]

            # compute losses per domain
            for idx, domain_name in enumerate(self.eval_domains):
                domain_idx = self.domain2idx[domain_name]
                domain_mask = (domain_ids == domain_idx).flatten()
                examplecounts[idx] = examplecounts[idx] + domain_mask.sum()

                if domain_mask.sum() > 0:
                    domain_labels = labels[domain_mask]
                    domain_preds = logits[domain_mask]
                    domain_labels = domain_labels[:, 1:].contiguous().view(-1)
                    domain_preds = domain_preds[:, :-1, :].contiguous().view(-1, domain_preds.size(-1))
                    losses[idx] = losses[idx] + loss_fn(domain_preds, domain_labels)
                    tokencounts[idx] = tokencounts[idx] + (domain_labels != -100).sum()
            
            if step >= self.max_eval_steps:
                break

        if self.args.world_size>1:
            torch.distributed.all_reduce(losses)
            torch.distributed.all_reduce(tokencounts)
            torch.distributed.all_reduce(examplecounts)

        # losses/preds/labels on CPU (final containers)
        per_domain_losses = {domain_name: losses[idx].item()
                             for idx, domain_name in enumerate(self.eval_domains) if tokencounts[idx] > 0}
        per_domain_tokencounts = {domain_name: tokencounts[idx].item()
                                  for idx, domain_name in enumerate(self.eval_domains) if tokencounts[idx] > 0}
        per_domain_examplecounts = {domain_name: examplecounts[idx].item()
                                    for idx, domain_name in enumerate(self.eval_domains) if tokencounts[idx] > 0}

        # normalize
        per_domain_losses = {domain_name: per_domain_losses[domain_name] / per_domain_tokencounts[domain_name]
                             for domain_name in per_domain_losses.keys()}

        metrics = {f"{domain_name}:log_perplexity": per_domain_losses[domain_name]
                   for domain_name in per_domain_losses.keys()}
        metrics["uniform_avg_log_ppl"] = np.mean(list(per_domain_losses.values()))
        metrics["worst_case_log_ppl"] = np.amax(list(per_domain_losses.values()))

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=sum(list(per_domain_examplecounts.values())))

    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [epoch_iterator.__next__()]
            except StopIteration:
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()

        return batch_samples, num_items_in_batch
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            print(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.state.init_training_references(self, train_dataloader, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                # epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            epoch_mix_tgt_iter = iter(self.tgt_loader)
            epoch_mix_train_iter = iter(self.train_loader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                
                lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 1e-4
                if self.reweight_train_iters > 0 and (self.state.global_step+1) % self.reweight_train_iters == 0:
                    self.domain_reweight_step(individual_train_loader=self.individual_train_loader,
                                              mix_tgt_iter=epoch_mix_tgt_iter,
                                              lr_value=lr_t)

                if self.reweight_tgt_iters > 0 and (self.state.global_step+1) % self.reweight_tgt_iters == 0:
                    self.tgt_reweight_step(individual_tgt_loader=self.individual_tgt_loader,
                                           mix_train_iter=epoch_mix_train_iter,
                                           lr_value=lr_t)
                    
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
                
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        # self.log(metrics)
        # self.accelerator.log()

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    
## test ##
if __name__ == '__main__':
    from data import DataTrainingArguments, get_train_eval_datasets, get_data_collator
    os.environ["WANDB_PROJECT"] = 'multi-target-reweight'
    data_config = DataTrainingArguments(dataset='slim_ood-logiqa-piqa-arc_easy-arc_challenge-hellaswag-sciq',
                                        max_train_samples=None,
                                        max_eval_samples=5000)
    individual_train_ds, individual_tgt_ds, individual_val_ds, domain_config, tokenizer = get_train_eval_datasets(data_config=data_config,
                                                            verbose=True,)
    data_collator=get_data_collator(tokenizer, do_padding=False, max_length=512)
    model_config = ModelArguments()
    rw_model_gpt2, rw_config = get_model_from_config(model_config, reweight=True)
    
    ## setup trainer ##
    # dist.init_process_group("nccl", rank=training_args.local_rank, world_size=training_args.world_size)
    
    fp16 = torch.cuda.is_available()
    training_args = FullTrainingArguments(output_dir="/scratch/homes/sfan/multi_doge/exp/test_run_2/",
                                    do_train=True,
                                    do_eval=True,
                                    learning_rate=1e-4,
                                    per_device_train_batch_size=4,
                                    warmup_steps=500,
                                    max_steps=1000,
                                    save_strategy='steps',
                                    save_steps=100,
                                    eval_strategy="steps",
                                    eval_steps=100,
                                    logging_steps=10,
                                    logging_strategy="steps",
                                    lr_scheduler_name='linear_warmup_decay',
                                    fp16=fp16,
                                    #   no_cuda=True,
                                    use_cpu=False,
                                    # ddp_backend='nccl',
                                    overwrite_output_dir=True,
                                    report_to=["wandb", "tensorboard"]
                                    )
    # Detecting last checkpoint.
    last_checkpoint = None
    num_skip_examples = 0
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            state = TrainerState.load_from_json(str(Path(last_checkpoint) / TRAINER_STATE_NAME))
            global_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
            num_skip_examples = state.global_step * global_batch_size
            logger.info(f"Skipping {num_skip_examples} examples")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # turn off find unused parameters
    training_args.ddp_find_unused_parameters = False
    # training_args.local_rank = -1
    print("training local_rank: ", training_args.local_rank)
    torch.cuda.empty_cache()
    # Initialize our Trainer
    trainer = MAPTrainer(
        model=rw_model_gpt2,
        args=training_args,
        domain_args=domain_config,
        train_dataset_ls=individual_train_ds,
        tgt_dataset_ls=individual_tgt_ds,
        val_dataset_ls=individual_val_ds,
        eval_dataset=individual_val_ds,
        processing_class=tokenizer,
        data_collator=get_data_collator(tokenizer, do_padding=data_config.do_padding, max_length=data_config.max_token_length),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    print('DoGE(MAP) launched! ❤️‍🔥❤️‍🔥')