### utils for reweighting
import os
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, TensorDataset

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")

@dataclass
class ToyDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dir: str = field(
        default='.', metadata={"help": "Path to the dataset directory."}
    )
    train_domains: str = field(
        default='D1,D2,D3,D4,D5,D6', metadata={"help": "domain names for training (D1, D2, ..., D6)."}
    )
    tgt_domains: str = field(
        default='L1,L2', metadata={"help": "target tasks (L1, ..., L4)."}
    )
    train_dw: str = field(
        default=None, metadata={"help": "training domain weights."}
    )
    tgt_dw: str = field(
        default=None, metadata={"help": "target task weights."}
    )
    val_dw: str = field(
        default=None, metadata={"help": "validation domain weights."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


def get_domain_dataset(domain_name, dataset_dir):
    data_path = os.path.join(dataset_dir, domain_name)
    train_data_path = os.path.join(data_path, "train.pt")
    val_data_path = os.path.join(data_path, "val.pt")
    train_dataset = torch.load(train_data_path, weights_only=False)
    if os.path.exists(val_data_path):
        val_dataset = torch.load(val_data_path, weights_only=False)
        return train_dataset, val_dataset
    return train_dataset, None


def load_data(data_config):
    assert data_config.train_domains is not None, "Training domains undefined!"
    assert data_config.tgt_domains is not None, "Target domains undefined!"
    train_domains = data_config.train_domains.split(",")
    tgt_domains = data_config.tgt_domains.split(",")
    
    # Mapping from domain name to dataset
    train_dataset_ls, tgt_dataset_ls, val_dataset_ls = [], [], []
    for dom in train_domains:
        train_ds, _ = get_domain_dataset(domain_name=dom, dataset_dir=DATASET_DIR)
        train_dataset_ls.append(train_ds)
    
    for dom in tgt_domains:
        assert dom not in train_domains, "Don't directly train on target function set!"
        tgt_ds, val_ds = get_domain_dataset(domain_name=dom, dataset_dir=DATASET_DIR)
        tgt_dataset_ls.append(tgt_ds)
        val_dataset_ls.append(val_ds)
    return train_dataset_ls, tgt_dataset_ls, val_dataset_ls


class WeightedDataLoader:
    def __init__(self, dataset_ls, weights,
                 dataset_name: list = None,
                 batch_size: int = 64, 
                 num_worker: int = 4):
        assert len(dataset_ls) == len(weights), f"Length of the dataset list({len(dataset_ls)}) and sampling weights({len(weights)}) not match!"
        self.dataset_ls = dataset_ls
        self.dataset_name = dataset_name
        self.weights = weights
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.get_weighted_dataloader()
    
    def get_weighted_dataloader(self):
        self.instance_weights = np.concatenate([np.asarray([w / len(d)]*len(d)) for w,d in zip(self.weights, self.dataset_ls)])
        combined_dataset = ConcatDataset(self.dataset_ls)
        self.sampler = WeightedRandomSampler(weights=self.instance_weights, num_samples=len(combined_dataset))
        self.data_loader = DataLoader(
            combined_dataset, batch_size=self.batch_size, num_workers=self.num_worker, sampler=self.sampler, drop_last=True)
        self.data_iter = iter(self.data_loader)
        print(f"log: DataLoader updated with sampling weights {self.weights} 🪄")
        
    def update_weights(self, new_weights):
        self.weights = new_weights
        self.get_weighted_dataloader()
        
    def __iter__(self):
        try:
            yield next(self.data_iter)
        except StopIteration:
            self.get_weighted_dataloader()
            yield next(self.data_iter)


class IndividualDataLoader:
    def __init__(self, dataset_ls, 
                 dataset_name: list = None,
                 batch_size: int = 64, 
                 num_worker: int = 4):
        self.dataset_ls = dataset_ls
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.get_individual_dataloader()
    
    def get_individual_dataloader(self):
        self.loader_ls = [DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_worker, drop_last=True) for ds in self.dataset_ls]
        self.data_iters = [iter(dl) for dl in self.loader_ls]
        
    def __iter__(self):
        try:
            yield [next(data_iter) for data_iter in self.data_iters]
        except StopIteration:
            self.get_individual_dataloader()
            yield [next(data_iter) for data_iter in self.data_iters]

     
def interleave_dataloader(dataset_ls, weights,
                          batch_size: int = 64,
                          num_worker: int = 4):
    assert len(dataset_ls) == len(weights), "Length of the dataset list and sampling weights not match!"
    data_loader = WeightedDataLoader(dataset_ls=dataset_ls,
                                     weights=weights,
                                     batch_size=batch_size,
                                     num_worker=num_worker)
    return data_loader


def individual_dataloader(dataset_ls, 
                          batch_size: int = 64,
                          num_worker: int = 1,):
    data_loader = IndividualDataLoader(dataset_ls=dataset_ls,
                                       batch_size=batch_size,
                                       num_worker=num_worker)
    return data_loader


if __name__ == '__main__':
    train_weights = [0.6, 0.3, 0.1]
    tgt_weights = [0.5, 0.5]
    valid_weights = [0.2,0.2]
    data_config = ToyDataArguments()
    train_dataset_ls, tgt_dataset_ls, val_dataset_ls = load_data(data_config)
    train_loader = interleave_dataloader(train_dataset_ls, train_weights,
                          batch_size = 64,
                          num_worker = 0)
    tgt_loader = interleave_dataloader(tgt_dataset_ls, tgt_weights,
                          batch_size = 64,
                          num_worker = 0)
    valid_loader = interleave_dataloader(val_dataset_ls, valid_weights,
                          batch_size = 64,
                          num_worker = 0)
    
    for x,y in train_loader:
        print(x)
        print(y)
        break
    import pdb
    pdb.set_trace()
    
    
