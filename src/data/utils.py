import numpy as np
from typing import Dict
import random
import itertools

from .slim_redpajama import get_slim_redpajama, get_slim_redpajama_6b, SUBSET2META
from .wiki40b import get_wiki40b
from .benchmarks import SUPPORTED_TASK_MAP
from .benchmarks import *
import numpy as np


def get_benchmark_data(dataset_name:str = "arc_easy",
                       num_proc: int = 10) -> Dict[str, np.ndarray]:
    """
    Dynamically calls the dataset_name - corresponding function to return data dict 
    """
    assert dataset_name in SUPPORTED_TASK_MAP, f"Module for dataset '{dataset_name}' not found."
    data_func = SUPPORTED_TASK_MAP[dataset_name]
    return data_func(num_proc=num_proc)

    
def get_dataset(args, dataset=None) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if dataset is not None:
        trg_dataset = dataset
    else:
        trg_dataset = args.dataset
    print(f"Loading train dataset '{trg_dataset}'")

    if 'wiki40b' in trg_dataset:
        lang_list = ['en', 'ar', 'zh', 'nl', 'fr', 'de', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'es', 
                     'th', 'tr', 'bg', 'ca', 'cs', 'da', 'el', 'et', 'fa', 'fi', 'he', 'hi', 'hr', 'hu', 'id', 
                     'lt', 'lv', 'ms', 'no', 'ro', 'sk', 'sl', 'sr', 'sv', 'tl', 'uk', 'vi']
        subset_list = trg_dataset.split('-')[1:]
        rst_dict = {}
        rst_dict['train'] = {}
        rst_dict['val'] = {}
        for subset in subset_list:
            if subset not in lang_list:
                continue
            if subset == "zh":
                subset = "zh-cn"
            subset_data = get_wiki40b(subset=subset, num_proc=10)
            rst_dict['train'][subset] = subset_data['train']
            rst_dict['val'][subset] = subset_data['val']
            print(f"Subset {subset}: train[{len(subset_data['train'])}]|val[{len(subset_data['val'])}]")
        return rst_dict
    
    elif 'slim_6b' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset == 'all' or args.eval_all_domains:
            all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            for k in SUBSET2META.keys():
                subset_data = get_slim_redpajama_6b(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val']
                all_train_list.append(subset_data['train'])
                all_val_list.append(subset_data['val'])
            train_data = np.concatenate(all_train_list)
            val_data = np.concatenate(all_val_list)
            rst_dict['train']['all'] = train_data
            rst_dict['val']['all'] = val_data
            
            if subset != 'all':
                rst_dict['train'] = rst_dict['train'][subset]
                if 'all' in rst_dict['val'].keys():
                    rst_dict['val'].pop('all')
            return rst_dict
        return get_slim_redpajama_6b(subset=subset, num_proc=10)
    
    elif 'slim_ood' in trg_dataset:
        subset_ood = trg_dataset.split('-')[1:]
        rst_dict = {}
        rst_dict['train'] = {}
        rst_dict['val'] = {}
        n_items_train = 10000000000
        n_items_train_ood = 50000000
        n_items_val = 5000
        
        for k in SUBSET2META.keys():
            subset_data = get_slim_redpajama(subset=k, num_proc=10)
            if k in subset_ood:
                rst_dict['train'][k] = subset_data['train'][:min(n_items_train,len(subset_data['train']))]
                rst_dict['val'][k] = subset_data['val'][:n_items_val*args.max_token_length]
            else:
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val'][:n_items_val*args.max_token_length]
        
        for s in subset_ood:
            if s in rst_dict['train'].keys():
                continue
            try:
                subset_data = get_benchmark_data(dataset_name=s)
            except:
                print(f"Dataset: {s} is not implemented!")
                continue
            rst_dict['train'][s] = subset_data['train'][:n_items_train]
            rst_dict['val'][s] = subset_data['val'][:n_items_val*args.max_token_length]
        # print(rst_dict['train'].keys())
        return rst_dict
    
    elif 'slim_full' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset =='all':
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            n_items_mix_train = (2000000000//args.max_token_length)//7
            n_items_mix_val = 12800
            
            for k in SUBSET2META.keys():
                subset_data = get_slim_redpajama(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train'][:-n_items_mix_train*args.max_token_length]
                rst_dict['val'][k] = subset_data['val'][:n_items_mix_val*args.max_token_length]
            return rst_dict
        
        elif subset == 'mix':
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            mix_data_train = []
            mix_data_val = []
            n_items_mix_train = (2000000000//args.max_token_length)//7
            n_items_mix_val = 2000
            
            for k in SUBSET2META.keys():
                subset_data = get_slim_redpajama(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train'][:-n_items_mix_train*args.max_token_length]
                rst_dict['val'][k] = subset_data['val'][:n_items_mix_val*args.max_token_length]
                mix_data_train.append(subset_data['train'][-n_items_mix_train*args.max_token_length:])
                mix_data_val.append(subset_data['val'][-n_items_mix_val*args.max_token_length:])
            
            mix_train_data = np.concatenate(mix_data_train)
            mix_val_data = np.concatenate(mix_data_val)
            # shuffle
            A = np.arange(0, len(mix_train_data), args.max_token_length)
            np.random.shuffle(A)
            mix_train_data = np.concatenate([mix_train_data[i:i+args.max_token_length] for i in A])
            
            B = np.arange(0, len(mix_val_data), args.max_token_length)
            np.random.shuffle(B)
            mix_val_data = np.concatenate([mix_val_data[i:i+args.max_token_length] for i in B])
            
            rst_dict['train']['mix'] = mix_train_data
            rst_dict['val']['mix'] = mix_val_data
            return rst_dict
        return get_slim_redpajama(subset=subset, num_proc=10)
    else:
        raise NotImplementedError(f"Unknow dataset key '{trg_dataset}'")
