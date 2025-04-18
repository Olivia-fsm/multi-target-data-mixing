#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np

def setup_environment(base_dir="/mloscratch/homes/glarou"):
    env_vars = {
        "HF_DATASETS_CACHE": f"{base_dir}/hf_datasets_cache",
        "HF_HOME": f"{base_dir}/hf_home",
    }
    
    for key, path in env_vars.items():
        os.environ[key] = path
        os.makedirs(path, exist_ok=True)

    return env_vars

env_vars = setup_environment()

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)
    
def extract_model_name(model_path):
    path_parts = model_path.split(os.path.sep)
    name = "-".join(path_parts[-2].split('-')[0:2])    
    return name

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a set of LM‑Eval tasks against any HF checkpoint"
    )
    p.add_argument(
        "--model_path", "-m",
        required=True,
        help="Path or HF repo ID for your model checkpoint"
    )
    p.add_argument(
        "--tasks", "-t",
        default="mathqa,bigbio_medqa,ai2_arc,piqa,sciq,hellaswag,logiqa",
        help="Comma‑separated list of task names to run"
    )
    p.add_argument(
        "--device", "-d",
        default="cuda",
        help="torch device (cpu|cuda)"
    )
    p.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    p.add_argument(
        "--num_fewshot", "-n",
        type=int,
        default=0,
        help="Number of few‑shot examples"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max examples per task (omit for full run)"
    )
    p.add_argument(
        "--include_path",
        default=None,
        help="Additional folder of custom task YAMLs"
    )
    p.add_argument(
        "--output_directory", "-o",
        default="evaluation_results",
        help="Directory to write results"
    )
    return p.parse_args()

def extract_accuracy_metrics(results):
    """Extract just the accuracy metrics from results"""
    accuracy_results = {}
    for task, metrics in results["results"].items():
        accuracy_results[task] = {
            metric.replace(",none", ""): value 
            for metric, value in metrics.items() 
            if "acc" in metric
        }
    return accuracy_results

def calculate_average_accuracy(accuracy_results):
    """Calculate the average acc_norm across all tasks"""
    acc_norm_values = []
    
    for task, metrics in accuracy_results.items():
        for metric, value in metrics.items():
            if metric == "acc_norm":
                if hasattr(value, 'item'):
                    value = value.item()
                acc_norm_values.append(float(value))
    
    if acc_norm_values:
        return sum(acc_norm_values) / len(acc_norm_values)
    return None

def print_summary(model_path, accuracy_results):
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Model: {model_path}")
    
    for task, metrics in accuracy_results.items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, str):
                print(f"  {metric}: {value}")
            else:
                value = value.item() if hasattr(value, 'item') else float(value)
                print(f"  {metric}: {value:.4f}")
    
    # Calculate and print average acc_norm
    avg_acc_norm = calculate_average_accuracy(accuracy_results)
    if avg_acc_norm is not None:
        print(f"\n===== AVERAGE PERFORMANCE =====")
        print(f"Average acc_norm across all tasks: {avg_acc_norm:.4f}")

def main():
    args = parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    
    tm = TaskManager(include_path=args.include_path) if args.include_path else TaskManager()

    lm = HFLM(
        pretrained=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        trust_remote_code=True,
        use_auth_token=False,
    )

    # Run evaluation
    results = simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        task_manager=tm
    )

    accuracy_results = extract_accuracy_metrics(results)
    
    avg_acc_norm = calculate_average_accuracy(accuracy_results)
    
    # Prepare output
    output_dict = {
        "results": accuracy_results,
        "average_acc_norm": avg_acc_norm,
        "metadata": {
            "model_path": args.model_path,
            "tasks": args.tasks,
            "num_fewshot": args.num_fewshot,
            "batch_size": args.batch_size,
        }
    }
    
    output_file = f"{args.output_directory}/{extract_model_name(args.model_path)}.json"
    with open(output_file, "w") as f:
        json.dump(output_dict, f, cls=NumpyEncoder, indent=2)
    
    print_summary(args.model_path, accuracy_results)
    print(f"\nResults saved as JSON to {output_file}")

if __name__ == "__main__":
    main()
