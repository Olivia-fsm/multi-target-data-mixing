import os
import json
import numpy as np
import argparse
from collections import defaultdict

all_metrics = {
    "gsm8k": "GSM8K",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "sciq": "SciQ",
    "kodcode": "KodCode",
    "logiqa": "LogiQA",
    "mathqa": "MathQA",
    "medqa": "MedQA",
}

dataset_subsets = {
    'redpj': ["arxiv", "book", "c4", "cc", "github", "stackexchange", "wikipedia"],
    'climblab': [f"cluster_{i}" for i in range(1, 21)],
    'wiki40b': ["en", "fr", "de", "es", "it", "ru"]
}

# Define ablation configurations
ablation_configs = {
    'redpj':{
        'T1': {
            'name': 'gsm8k+arc_easy+arc_challenge',
            'metrics': ['gsm8k', 'arc_easy', 'arc_challenge']
        },
        'T2': {
            'name': 'gsm8k+hellaswag',
            'metrics': ['gsm8k', 'hellaswag']
        },
        'T3': {
            'name': 'gsm8k+piqa',
            'metrics': ['gsm8k', 'piqa']
        },
        'T4': {
            'name': 'gsm8k+logiqa',
            'metrics': ['gsm8k', 'logiqa']
        },
        'T5': {
            'name': 'gsm8k+sciq',
            'metrics': ['gsm8k', 'sciq']
        },
        'T6': {
            'name': 'gsm8k+kodcode+arc_easy+arc_challenge',
            'metrics': ['gsm8k', 'kodcode', 'arc_easy', 'arc_challenge']
        },
        'T7': {
            'name': 'gsm8k+kodcode+hellaswag',
            'metrics': ['gsm8k', 'kodcode', 'hellaswag']
        },
        'T8': {
            'name': 'all_tasks',
            'metrics': list(all_metrics.keys())
        },
        'T9': {
            'name': '6_tasks_without_kodcode_gsm8k',
            'metrics': ['sciq', 'arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'logiqa']
        },
        'T10': {
            'name': 'reasoning + mathqa + medqa',
            'metrics': ['sciq', 'arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'logiqa', 'mathqa', 'medqa']
        },
        'T11': {
            'name': 'arc + mathqa',
            'metrics': ['arc_easy', 'arc_challenge', 'medqa', 'mathqa']
        },
        'T12': {
            'name': 'hellaswag + logiqa + medqa + mathqa',
            'metrics': ['medqa', 'mathqa', 'hellaswag', 'logiqa']
        },
    },
    'wiki40b': {
        'T1': {
            'name': 'uk+ca+da+ro',
            'metrics': ['uk', 'ca', 'da', 'ro']
        },
        'T2': {
            'name': 'pl+uk+nl+pt+ca+tr+da+ro',
            'metrics': ['pl', 'uk', 'nl', 'pt', 'ca', 'tr', 'da', 'ro']
        },
    },
    "climblab": {
        'T1': {
            'name': 'gsm8k+arc_easy+arc_challenge',
            'metrics': ['gsm8k', 'arc_easy', 'arc_challenge']
        },
        'T2': {
            'name': 'gsm8k+hellaswag',
            'metrics': ['gsm8k', 'hellaswag']
        },
        'T3': {
            'name': 'gsm8k+piqa',
            'metrics': ['gsm8k', 'piqa']
        },
        'T4': {
            'name': 'gsm8k+logiqa',
            'metrics': ['gsm8k', 'logiqa']
        },
        'T5': {
            'name': 'gsm8k+sciq',
            'metrics': ['gsm8k', 'sciq']
        },
        'T6': {
            'name': 'gsm8k+kodcode+arc_easy+arc_challenge',
            'metrics': ['gsm8k', 'kodcode', 'arc_easy', 'arc_challenge']
        },
        'T7': {
            'name': 'gsm8k+kodcode+hellaswag',
            'metrics': ['gsm8k', 'kodcode', 'hellaswag']
        },
        'T8': {
            'name': 'all_tasks',
            'metrics': list(all_metrics.keys())
        },
        'T9': {
            'name': '6_tasks_without_kodcode_gsm8k',
            'metrics': ['sciq', 'arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'logiqa']
        },
        'T10': {
            'name': 'reasoning + mathqa + medqa',
            'metrics': ['sciq', 'arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'logiqa', 'mathqa', 'medqa']
        },
        'T11': {
            'name': 'arc + mathqa',
            'metrics': ['arc_easy', 'arc_challenge', 'medqa', 'mathqa']
        },
        'T12': {
            'name': 'hellaswag + logiqa + medqa + mathqa',
            'metrics': ['medqa', 'mathqa', 'hellaswag', 'logiqa']
        },
    },
}

def load_weights(weights_file):
    """Load domain weights from file"""
    if not os.path.exists(weights_file):
        return None
         
    try:
        with open(weights_file, 'r') as f:
            domain_weights = json.load(f)
        
        for domain, sources in domain_weights.items():
            domain_weights[domain] = {k: float(v) for k, v in sources.items()}
        
        return domain_weights
    except Exception:
        return None

def process_domain_weights(weights_file, config, dataset):
    """Process domain weights and return normalized weights string"""
    try:
        domain_weights = load_weights(weights_file)
        if not domain_weights:
            return None
        
        domains_to_process = config.get('metrics', [])
        train_domains = dataset_subsets[dataset]
        
        # Sum weights for each domain
        summed_weights = {domain: 0.0 for domain in train_domains}
        count = 0
        
        for task_domain in domains_to_process:
            if task_domain in domain_weights:
                count += 1
                for domain in train_domains:
                    if domain in domain_weights[task_domain]:
                        summed_weights[domain] += float(domain_weights[task_domain][domain])
        
        # Average and normalize weights
        if count > 0:
            averaged_weights = {domain: summed_weights[domain] / count for domain in train_domains}
            total_weight = sum(averaged_weights.values())
            
            if total_weight > 0:
                normalized_weights = {domain: weight / total_weight for domain, weight in averaged_weights.items()}
                # normalize, round to 4 decimal places and convert to string
                weights_str = ",".join([str(round(normalized_weights[domain], 4)) for domain in train_domains])
                # weights_str = ",".join([str(normalized_weights[domain]) for domain in train_domains])
                return weights_str
        
        return None
    except Exception:
        return None

def generate_training_config(config_id, config, weights_file, dataset):
    """Generate training configuration for a specific ablation setting"""
    train_domains = dataset_subsets[dataset]
    train_dw = process_domain_weights(weights_file, config, dataset)
    if dataset == "redpj" or "slimpajama":
        dataset_name = "slim_ood"
    else:
        dataset_name = dataset
    json_config = {
        "dataset": f"{dataset_name}-{'-'.join(config['metrics'])}" if (dataset == 'redpj' or  dataset == 'climblab') else f"{dataset}-{'-'.join(config['metrics'])}-{'-'.join(train_domains)}",
        "train_domains": ",".join(train_domains),
        "tgt_domains": ','.join(config['metrics']),
        "max_steps": 20000,
        "train_dw": train_dw,
        "tgt_dw": None,
        "val_dw": None,
        "max_train_samples": None,
        "max_eval_samples": 5000,
        "max_token_length": 512,
        "seed": 16,
        "preprocessing_num_workers": 2,
        "model_name_or_path": None,
        "model_type": "gpt2",
        "config_overrides": "n_positions=512,n_embd=768,n_layer=12,n_head=12",
        "run_name": f"CRISP-125M",
        "output_dir": "/mloscratch/homes/glarou/DoGE/regmix/multi_doge/exp",
        "do_train": True,
        "do_eval": True,
        "do_predict": False,
        "learning_rate": 0.0005,
        "weight_decay": 0.01,
        "reweight_train": "None",
        "reweight_tgt": "None",
        "reweight_train_iters": 0,
        "reweight_tgt_iters": 0,
        "ref_model": None,
        "lr_scheduler_name": "linear_warmup_cosine",
        "lr_end": 0.0001,
        "reweight_eps": 0.0,
        "mu_train": 0.001,
        "mu_tgt": 0.0002,
        "max_grad_norm": 5.0,
        "per_device_train_batch_size": 16,
        "warmup_ratio": 0.05,
        "warmup_steps": 500,
        "save_steps": 5000,
        "eval_steps": 500,
        "gradient_accumulation_steps": 2,
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 10,
        "ddp_find_unused_parameters": False,
        "use_cpu": False,
        "compute_pertoken_losses": False,
        "overwrite_output_dir": False
    }
    
    return json_config

def main(dataset='redpj', weights_dir="./crisp_results"):
    dataset_weights_dir = os.path.join(weights_dir, dataset)
    weights_file = os.path.join(dataset_weights_dir, "specialist_subset_distributions_k1_5_k2_5.json")
    output_dir = os.path.join(dataset_weights_dir, "configs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(weights_file):
        print(f"Error: Weights file not found at {weights_file}")
        return
    
    for config_id, config in ablation_configs[dataset].items():
        training_config = generate_training_config(config_id, config, weights_file, dataset)
        with open(os.path.join(output_dir, f"{config_id}-crisp.json"), "w") as f:
            json.dump(training_config, f, indent=4)
    
    print(f"Generated {len(ablation_configs[dataset])} configurations for {dataset} from {weights_file}")
    print(f"Configuration files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate training configurations')
    parser.add_argument('--dataset', type=str, default='climblab',
                        help='Dataset to use (redpj or wiki40b or climblab)')
    parser.add_argument('--weights_dir', type=str, default="./crisp_results",
                        help='Base directory containing weight files')
    
    args = parser.parse_args()
    main(dataset=args.dataset, weights_dir=args.weights_dir)
