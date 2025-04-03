## Data Arguments ##
from dataclasses import dataclass, field
import pickle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import TrainingArguments, MODEL_FOR_CAUSAL_LM_MAPPING, DataCollatorWithPadding
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import sys
sys.path.append("/scratch/homes/sfan/multi_doge/src")
from data.utils import get_dataset
import random
import numpy as np
from datasets import Dataset, IterableDataset, load_from_disk, concatenate_datasets, interleave_datasets
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from pathlib import Path
from collections import Counter
from copy import deepcopy

import transformers
from transformers import AutoTokenizer
from transformers import TrainingArguments, MODEL_FOR_CAUSAL_LM_MAPPING

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
# from data.weighted_dataloader import WeightedDataLoader
# transformers.utils.move_cache('/mloraw1/sfan/huggingface_cache')


RANDOM_BATCH_SIZE = 512
DEFAULT_SEED=111


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dir: str = field(
        default='.', metadata={"help": "Path to the dataset directory."}
    )
    dataset: str = field(
        default='redpajama-all', metadata={"help": "Name of the dataset."}
    )
    train_domains: str = field(
        default='arxiv,book,cc,c4,github,stackexchange,wikipedia', metadata={"help": "domain names for training."}
    )
    tgt_domains: str = field(
        default='logiqa,piqa,arc_easy,arc_challenge,hellaswag,sciq,kodcode,gsm8k', metadata={"help": "target domains for generalization."}
    )
    val_domains: str = field(
        default=None, metadata={"help": "domains for evauation."}
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
    max_token_length: int = field(
        default=512,
        metadata={
            "help": (
                "Input sequence length after tokenization. "
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    do_padding: bool = field(
        default=True, metadata={"help": "Pad the inputs."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Shuffle the training data on the fly"}
    )

@dataclass
class DomainConfigArguments:
    """
    Domain config settings. """
    domain_list: list = field(
        default_factory=list,
        # default=['arxiv', 'book', 'c4', 'cc', 'github', 'stackexchange', 'wikipedia'], 
        metadata={"help": "List of domain names."}
    )
    train_dw: torch.Tensor = field(
        default=None, 
        metadata={"help": "Training domain weights."}
    )
    tgt_dw: torch.Tensor = field(
        default=None, 
        metadata={"help": "Target task weights."}
    )
    val_dw: torch.Tensor = field(
        default=None, 
        metadata={"help": "Validation domain weights."}
    )
    idx2domain: dict = field(
        default_factory=dict,
        metadata={"help": "index mapping to domain names."}
    )
    domain2idx: dict = field(
        default_factory=dict, 
        metadata={"help": "domain names mapping to indices."}
    )
    train_ids: torch.Tensor = field(
        default=None, 
        metadata={"help": "Training domain indices in `domain_list`."}
    )
    tgt_ids: torch.Tensor = field(
        default=None, 
        metadata={"help": "Target domain indices in `domain_list`."}
    )
    val_ids: torch.Tensor = field(
        default=None, 
        metadata={"help": "Validation domain indices in `domain_list`."}
    )

def domain_gen(data, seq_len, domain_ids=None):
    """Construct generator for domain datasets."""
    if domain_ids is None:
        for i in range(len(data)//seq_len):
            yield {"input_ids": data[i*seq_len:(i+1)*seq_len]}
    else:
        for i in range(len(data)//seq_len):
            yield {"domain_ids": torch.tensor([domain_ids], dtype=torch.long), "input_ids": data[i*seq_len:(i+1)*seq_len]}

# import torch
# from torch.utils.data import DataLoader, Sampler
# import numpy as np
# from typing import List, Callable, Optional, Iterator
# from itertools import chain

# class WeightedDataset(torch.utils.data.Dataset):
#     """
#     A dataset that combines multiple datasets with their respective lengths
#     """
#     def __init__(self, datasets: List[Dataset]):
#         self.datasets = datasets
#         self.lengths = [len(dataset) for dataset in datasets]
#         self.cumsum_lengths = np.cumsum([0] + self.lengths)
    
#     def __len__(self) -> int:
#         return sum(self.lengths)
    
#     def __getitem__(self, idx: int):
#         # Find which dataset the index belongs to
#         dataset_idx = np.searchsorted(self.cumsum_lengths[1:], idx, side='right')
#         # Get the local index within that dataset
#         local_idx = idx - self.cumsum_lengths[dataset_idx]
#         return self._get_dataset_item(self.datasets[dataset_idx], local_idx)
    
#     def _get_dataset_item(self, local_dataset, local_idx: int):
#         return {k: local_dataset[k][local_idx] for k in local_dataset.features}

# class WeightedRandomSampler(Sampler[int]):
#     """
#     Samples elements from [0,...,len(weights)-1] with given probabilities (weights)
#     Supports sampling more than 2^24 samples by chunking
#     """
#     def __init__(self, weights: List[float], lengths: List[int], num_samples: Optional[int] = None):
#         if not isinstance(weights, torch.Tensor):
#             weights = torch.tensor(weights, dtype=torch.float64)
        
#         if len(weights) != len(lengths):
#             raise ValueError("Number of weights must match number of datasets")
            
#         self.weights = weights / weights.sum()  # Normalize weights
#         self.lengths = lengths
#         self.num_samples = num_samples if num_samples is not None else sum(lengths)
        
#         # Create cumulative lengths for mapping global indices to local indices
#         self.cumsum_lengths = np.cumsum([0] + lengths)
        
#         # Maximum number of samples for torch.multinomial (2^24)
#         self.MAX_SAMPLES = 2**24 - 1
        
#     def __iter__(self) -> Iterator[int]:
#         remaining_samples = self.num_samples
        
#         while remaining_samples > 0:
#             # Calculate number of samples for this chunk
#             chunk_size = min(remaining_samples, self.MAX_SAMPLES)
            
#             # Sample dataset indices according to weights for this chunk
#             dataset_indices = torch.multinomial(
#                 self.weights,
#                 chunk_size,
#                 replacement=True
#             )
            
#             # For each selected dataset in this chunk, sample a random index
#             for dataset_idx in dataset_indices:
#                 start_idx = self.cumsum_lengths[dataset_idx]
#                 dataset_length = self.lengths[dataset_idx]
#                 # Generate random index within the selected dataset
#                 local_idx = torch.randint(dataset_length, (1,)).item()
#                 # Convert to global index
#                 yield start_idx + local_idx
            
#             remaining_samples -= chunk_size
    
#     def __len__(self) -> int:
#         return self.num_samples

#     def update_weights(self, new_weights: List[float]) -> None:
#         """
#         Update sampling weights
#         Args:
#             new_weights: New sampling weights for each dataset
#         """
#         if len(new_weights) != len(self.lengths):
#             raise ValueError("Number of new weights must match number of datasets")
        
#         if not isinstance(new_weights, torch.Tensor):
#             new_weights = torch.tensor(new_weights, dtype=torch.float64)
            
#         self.weights = new_weights / new_weights.sum()  # Normalize new weights

# class WeightedDataLoader(DataLoader):
#     """
#     DataLoader that performs weighted sampling across multiple datasets
#     """
#     def __init__(
#         self,
#         dataset_ls: List[Dataset],
#         weights: List[float],
#         batch_size: int,
#         data_collator: Optional[Callable] = None,
#         num_samples: Optional[int] = None,
#         **kwargs
#     ):
#         # Combine datasets
#         dataset = WeightedDataset(dataset_ls)
        
#         # Create sampler
#         self.sampler = WeightedRandomSampler(
#             weights=weights,
#             lengths=[len(ds) for ds in dataset_ls],
#             num_samples=num_samples
#         )
        
#         # Initialize parent DataLoader
#         super().__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             sampler=self.sampler,
#             collate_fn=data_collator,
#             **kwargs
#         )
        
#         # Store number of datasets for validation
#         self.num_datasets = len(dataset_ls)
    
#     def update_weights(self, new_weights: List[float]) -> None:
#         """
#         Update the sampling weights for each dataset
#         Args:
#             new_weights: List of new sampling weights for each dataset
#         """
#         if len(new_weights) != self.num_datasets:
#             raise ValueError(f"Expected {self.num_datasets} weights, got {len(new_weights)}")
#         self.sampler.update_weights(new_weights)
        
## Old DataLoader ##
class WeightedDataLoader:
    def __init__(self, dataset_ls, weights,
                 dataset_name: list = None,
                 batch_size: int = 64, 
                 num_worker: int = 4,
                 data_collator = None,
                 repeat = True,
                 **dataloader_params):
        assert len(dataset_ls) == len(weights), f"Length of the dataset list({len(dataset_ls)}) and sampling weights({len(weights)}) not match!"
        # self.get_dataset_chunk(dataset_ls)
        self.dataset_ls = dataset_ls
        self.dataset_name = dataset_name
        self.weights = weights
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.data_collator = data_collator
        
        if "batch_size" in dataloader_params:
            self.batch_size = dataloader_params.pop("batch_size")
        if "num_workers" in dataloader_params:
            self.num_workers = dataloader_params.pop("num_workers")
        if "drop_last" in dataloader_params:
            self.drop_last = dataloader_params.pop("drop_last")
        if "collate_fn" in dataloader_params:
            self.collate_fn = dataloader_params.pop("collate_fn")
        
        # combined_dataset = WeightedDataset(self.dataset_ls)
        # # weights: List[float], lengths: List[int]
        # self.sampler = WeightedRandomSampler(weights=self.weights, lengths = [len(ds) for ds in self.dataset_ls])
        self.get_dataset_chunk()
        self.get_weighted_dataloader(**dataloader_params)
    
    def get_dataset_chunk(self):
        MAX_CHUNK_SIZE = 1000000
        self.chunk_size = [min(MAX_CHUNK_SIZE, len(ds)) for ds in self.dataset_ls]
        self.dataset = concatenate_datasets([ds.select(np.random.choice(len(ds), cs, replace=False)) for ds,cs in zip(self.dataset_ls, self.chunk_size)])
        return self.dataset
    
    def get_weighted_dataloader(self,**dataloader_params):
        # combined_dataset = concatenate_datasets(self.dataset_ls)
        self.instance_weights = np.concatenate([np.asarray([w / csz]*csz) for w,csz in zip(self.weights, self.chunk_size)])
        self.sampler = WeightedRandomSampler(weights=self.instance_weights, num_samples=len(self.dataset))
        
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_worker, sampler=self.sampler, drop_last=True, collate_fn=self.data_collator, **dataloader_params)
        
        self.data_iter = iter(self.data_loader)
        # print(f"log: DataLoader updated with sampling weights {self.weights} 🪄")
        
    def update_weights(self, new_weights):
        if isinstance(new_weights, list):
            new_weights = np.asarray(new_weights)
        self.weights = new_weights / new_weights.sum()
        self.get_weighted_dataloader()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.get_dataset_chunk()
            self.get_weighted_dataloader()
            return next(self.data_iter)


class IndividualDataLoader:
    def __init__(self, dataset_ls, 
                 dataset_name: list = None,
                 batch_size: int = 64, 
                 num_workers: int = 4,
                 data_collator = None,
                 **dataloader_params):
        self.dataset_ls = dataset_ls
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_collator = data_collator
        
        if "batch_size" in dataloader_params:
            self.batch_size = dataloader_params.pop("batch_size")
        if "num_workers" in dataloader_params:
            self.num_workers = dataloader_params.pop("num_workers")
        if "drop_last" in dataloader_params:
            self.drop_last = dataloader_params.pop("drop_last")
        if "collate_fn" in dataloader_params:
            self.collate_fn = dataloader_params.pop("collate_fn")
            
        self.get_individual_dataloader(**dataloader_params)
    
    def get_individual_dataloader(self, **dataloader_params):
        self.loader_ls = [DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, collate_fn=self.data_collator, **dataloader_params) for ds in self.dataset_ls]
        self.data_iters = [iter(dl) for dl in self.loader_ls]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return [next(data_iter) for data_iter in self.data_iters]
        except StopIteration:
            self.get_individual_dataloader()
            return [next(data_iter) for data_iter in self.data_iters]

     
def interleave_dataloader(dataset_ls, weights,
                          batch_size: int = 64,
                          num_workers: int = 4,
                          data_collator = None,
                          **dataloader_params):
    assert len(dataset_ls) == len(weights), "Length of the dataset list and sampling weights not match!"
    data_loader = WeightedDataLoader(dataset_ls=dataset_ls,
                                     weights=weights,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     data_collator=data_collator,
                                     **dataloader_params)
    return data_loader


def individual_dataloader(dataset_ls, 
                          batch_size: int = 64,
                          num_workers: int = 1,
                          data_collator = None,
                          **dataloader_params):
    data_loader = IndividualDataLoader(dataset_ls=dataset_ls,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       data_collator=data_collator,
                                       **dataloader_params)
    return data_loader


def get_train_eval_datasets(data_config:DataTrainingArguments,
                            verbose:bool=False,
                            doremi:bool=False,
                            **kwargs):
    data_dict = get_dataset(data_config)
    if 'all' in data_dict['train'].keys():
        del data_dict['train']['all']
    if 'all' in data_dict['val'].keys():
        del data_dict['val']['all']
    if doremi and ('mix' in data_dict['train'].keys()):
        del data_dict['train']['mix']
        del data_dict['val']['mix']
        
    seed = 42
    sequence_length = data_config.max_token_length
    max_train_samples = data_config.max_train_samples
    max_eval_samples = data_config.max_eval_samples

    domain_list = list(data_dict['train'].keys())
    idx2domain = {i:dom for i,dom in enumerate(domain_list)}
    domain2idx = {dom:i for i,dom in idx2domain.items()}
    train_domains = data_config.train_domains.split(',') if data_config.train_domains else domain_list
    tgt_domains = data_config.tgt_domains.split(',') if data_config.tgt_domains else domain_list
    val_domains = data_config.val_domains.split(',') if data_config.val_domains else domain_list
    train_ids = torch.tensor([domain2idx[name] for name in train_domains])
    tgt_ids = torch.tensor([domain2idx[name] for name in tgt_domains])
    val_ids = torch.tensor([domain2idx[name] for name in val_domains])
    
    if data_config.train_dw is None:
        # sample target data seperately
        train_dw = torch.ones(len(train_ids), dtype=torch.float)/len(train_ids)
    else:
        train_dw = torch.tensor([float(i) for i in data_config.train_dw.split(",")])
    
    if data_config.tgt_dw is None:
        tgt_dw = torch.ones(len(tgt_ids), dtype=torch.float)/len(tgt_ids)
    else:
        tgt_dw = torch.tensor([float(i) for i in data_config.tgt_dw.split(",")])
    
    if data_config.val_dw is None:
        val_dw = torch.ones(len(val_ids), dtype=torch.float)/len(val_ids)
    else:
        val_dw = torch.tensor([float(i) for i in data_config.val_dw.split(",")])
    
    domain_config = DomainConfigArguments(
                    domain_list=domain_list, # include all train/tgt/val domains
                    idx2domain=idx2domain,
                    domain2idx=domain2idx,
                    train_ids=train_ids,
                    tgt_ids=tgt_ids,
                    val_ids=val_ids,
                    train_dw=train_dw,
                    tgt_dw=tgt_dw,
                    val_dw=val_dw,
                    **kwargs)
    
    train_dict = {dom:data_dict['train'][dom] for dom in train_domains if dom in data_dict['train']}
    tgt_dict = {dom:data_dict['train'][dom] for dom in tgt_domains if dom in data_dict['train']}
    val_dict = {dom:data_dict['val'][dom] for dom in val_domains if dom in data_dict['val']}
    
    train_dataset_ls, tgt_dataset_ls, val_dataset_ls = [], [], []
    for dom in train_domains:
        k = domain2idx[dom]
        train_domain_dataset = Dataset.from_generator(domain_gen,
                                                gen_kwargs={'data': train_dict[dom],
                                                            'seq_len': sequence_length,
                                                            'domain_ids': k,
                                                            }
                                                )
        train_dataset_ls.append(train_domain_dataset)
        if verbose:
            print(f'{idx2domain[k]} loaded!')
    
    for dom in tgt_domains:
        k = domain2idx[dom]
        tgt_domain_dataset = Dataset.from_generator(domain_gen,
                                                gen_kwargs={'data': tgt_dict[dom],
                                                            'seq_len': sequence_length,
                                                            'domain_ids': k,
                                                            }
                                                )
        tgt_dataset_ls.append(tgt_domain_dataset)
        if verbose:
            print(f'{idx2domain[k]} loaded!')
    
    for dom in val_domains:
        k = domain2idx[dom]
        val_domain_dataset = Dataset.from_generator(domain_gen,
                                                gen_kwargs={'data': val_dict[dom],
                                                            'seq_len': sequence_length,
                                                            'domain_ids': k,
                                                            }
                                                )
        val_dataset_ls.append(val_domain_dataset)

    def take_data_generator(ds, max_samples):
        idx = 0
        for ex in ds:
            yield ex
            idx += 1
            if max_samples is not None and idx >= max_samples:
                return
            
    if max_train_samples is not None:
        tgt_dataset_ls = [Dataset.from_generator(take_data_generator, gen_kwargs={'ds': ds, 'max_samples': max_train_samples}) for ds in tgt_dataset_ls]
        train_dataset_ls = [Dataset.from_generator(take_data_generator, gen_kwargs={'ds': ds, 'max_samples': max_train_samples}) for ds in train_dataset_ls]
        
    if max_eval_samples is not None:
        val_dataset_ls = [Dataset.from_generator(take_data_generator, gen_kwargs={'ds': ds, 'max_samples': max_eval_samples}) for ds in val_dataset_ls]
    if 'wiki40b' in data_config.dataset:
        tokenizer = AutoTokenizer.from_pretrained('/scratch/pagliard/doge/exp/doge_frozen_weights_12l_catalan-mu0001_seed42_10k/checkpoint-10000')
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    return train_dataset_ls, tgt_dataset_ls, val_dataset_ls, domain_config, tokenizer

def get_data_collator(tokenizer, return_tensors='pt', do_padding=False, max_length=1024):
    def data_collator(features):
        if not do_padding:
            try:
                batch = {
                        k: torch.tensor([f[k] for f in features])
                        for k in features[0].keys() if k!='input_ids'
                        }
                if not torch.is_tensor(batch['input_ids']):
                    pt_dtype = torch.int64
                    try:
                        if batch['input_ids'].dtype == 'uint16':
                            pt_dtype = torch.uint16
                        elif batch['input_ids'].dtype == 'int32':
                            pt_dtype = torch.int32
                        elif batch['input_ids'].dtype == 'int64':
                            pt_dtype = torch.int64
                    except:
                        pass
                    batch['input_ids'] = torch.tensor([np.array(f['input_ids']) for f in features], dtype=pt_dtype)
            except Exception:
                batch = {
                        k: torch.tensor([np.array(f[k]) for f in features])
                        for k in features[0].keys()
                        }
        else:
            try:
                batch = tokenizer.pad(features, return_tensors=return_tensors, pad_to_multiple_of=max_length)
            except:
                raise Exception
        # batch['input_ids'] = batch['input_ids'].long()
        if 'attention_mask' not in batch:
            batch['attention_mask'] = torch.ones_like(batch['input_ids']).long()
        else:
            batch['attention_mask'] = batch['attention_mask'].long()

        batch.pop("special_tokens_mask", None)
        if 'labels' not in batch:
            labels = batch['input_ids'].clone()
            batch["labels"] = labels

        if tokenizer.pad_token_id is not None:
            batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        # print(batch)
        return batch
    return data_collator 

# test #
if __name__ == '__main__':
    data_config = DataTrainingArguments(dataset='slim_ood-logiqa-piqa-arc_easy-arc_challenge-hellaswag-sciq-humaneval-gsm8k',
                                        max_train_samples=None,
                                        max_eval_samples=5000)
    individual_train_ds, individual_tgt_ds, individual_val_ds, domain_config, tokenizer = get_train_eval_datasets(data_config=data_config,
                                                            verbose=True,)
    data_collator=get_data_collator(tokenizer, do_padding=False, max_length=512)
    train_loader = interleave_dataloader(individual_train_ds, domain_config.train_dw,
                          batch_size = 4,
                          num_workers = 0,
                          data_collator=data_collator)
    tgt_loader = interleave_dataloader(individual_tgt_ds, domain_config.tgt_dw,
                          batch_size = 4,
                          num_workers = 0,
                          data_collator=data_collator)
    val_loader = individual_dataloader(individual_val_ds,
                          batch_size = 4,
                          num_workers = 0,
                          data_collator=data_collator)
    
    for train_batch in train_loader:
        print(train_batch)
        break
    
    for val_batch in val_loader:
        print(val_batch)
        break

    print("DataLoader Launched! 🦈")
    import pdb
    pdb.set_trace()