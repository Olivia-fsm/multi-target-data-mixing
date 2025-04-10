import os
import math
import warnings
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from transformers.utils import ExplicitEnum, is_torch_tpu_available
from transformers.optimization import get_scheduler

class LinearWarmupExponentialLR(LRScheduler):
    """
    Exponential LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # figure out decay rate to use to get within 1e-10 of lr_end at end of training
            gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_training_steps - self.num_warmup_steps))
                      for base_lr in self.base_lrs]
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for base_lr, gamma in zip(self.base_lrs, gammas)]


class LinearWarmupCosineLR(LRScheduler):
    """
    Cosine LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]


class LinearWarmupDecay(LRScheduler):
    """
    LR with linear warmup and linear decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False, decay_rate=0.1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_decay_steps = int(decay_rate * num_training_steps)
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        elif self.last_epoch < self.num_training_steps - self.num_decay_steps:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * (self.num_training_steps - self.last_epoch) / self.num_decay_steps for base_lr in self.base_lrs]

class LinearDecay(LRScheduler):
    """
    LR with linear warmup and linear decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False, decay_rate=1.0):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_decay_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_rate = decay_rate
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        # if os.environ.get("LOCAL_RANK", "0") == "0":
        #     import pdb
        #     pdb.set_trace()
        return [self.lr_end + (base_lr - self.lr_end) * (self.num_training_steps - self.last_epoch) / self.num_decay_steps for base_lr in self.base_lrs]

class ExtendedSchedulerType(ExplicitEnum):
    LINEAR_WARMUP_EXPONENTIAL = "linear_warmup_exponential"
    LINEAR_WARMUP_COSINE = "linear_warmup_cosine"
    LINEAR_WARMUP_DECAY = "linear_warmup_decay"
    LINEAR_DECAY = "linear_decay"


# extend scheduler function mapping
TYPE_TO_EXTENDED_SCHEDULER_FUNCTION = {
        ExtendedSchedulerType.LINEAR_WARMUP_EXPONENTIAL: LinearWarmupExponentialLR,
        ExtendedSchedulerType.LINEAR_WARMUP_COSINE: LinearWarmupCosineLR,
        ExtendedSchedulerType.LINEAR_WARMUP_DECAY: LinearWarmupDecay,
        ExtendedSchedulerType.LINEAR_DECAY: LinearDecay,
}


def get_scheduler_extended(
    name,
    optimizer,
    num_warmup_steps=0,
    num_training_steps=0,
    lr_end=1e-4,
    decay_rate=0.1,
):

    try:
        name = ExtendedSchedulerType(name)
        schedule_func = TYPE_TO_EXTENDED_SCHEDULER_FUNCTION[name]
    except ValueError:
        return get_scheduler(name, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if "decay" in name.lower():
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_end=lr_end, decay_rate=decay_rate)
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_end=lr_end)

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    