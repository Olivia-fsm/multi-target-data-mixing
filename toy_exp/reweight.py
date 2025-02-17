import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, TensorDataset
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from model import MLP
from data_utils import WeightedDataLoader, IndividualDataLoader, ToyDataArguments

## utils
def get_grad(model: nn.Module):
    # make sure all grads are detached #
    grads = []
    for p_name, p in model.named_parameters():
        flat_grad = p.grad.detach().flatten().cpu()
        grads.append(flat_grad)  
    flat_grad = torch.concat(grads)
    return flat_grad

def grad_align(tr_grad, tgt_grad):
    return tr_grad @ tgt_grad.T

def dro_reweight():
    return

def doge_reweight():
    return

