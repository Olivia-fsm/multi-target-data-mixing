import os
import subprocess
import numpy as np
import argparse
import re

# Directories
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Create argument parser
parser = argparse.ArgumentParser(description='Evaluate a model checkpoint on specific datasets')
parser.add_argument('--model_path', required=True, type=str, help='Path to your model checkpoint')
parser.add_argument('--device', default='cuda:0', type=str, help='Device to use for evaluation')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for evaluation')
parser.add_argument('--tasks', default="gsm8k,hellaswag,logiqa,ai2_arc,piqa,sciq,kodcode", 
                    type=str, help='Comma-separated list of tasks to evaluate')
parser.add_argument('--num_fewshot', default=5, type=int, help='Number of few-shot examples to use')

def parse_metrics(output):
    """
    Parse metrics from the lm_eval output using regex for better handling of special characters
    """
    metrics = {}
    task_results = {}
    
    task_match = re.search(r'=== Evaluating task: ([^\s]+) ===', output)
    current_task = task_match.group(1) if task_match else "unknown"
    
    # Define a regex pattern to match metric lines
    metric_pattern = r'\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|'
    
    # Find all matches
    last_task_name = None
    for match in re.finditer(metric_pattern, output):
        # Skip the header rows
        if match.group(1).strip() in ["Tasks", "-----"]:
            continue
        
        # Extract the fields from the match
        task_name = match.group(1).strip()
        
        # If task name is empty, use the last seen task name or current task
        if not task_name:
            task_name = last_task_name if last_task_name else current_task
        else:
            last_task_name = task_name
            
        filter_type = match.group(3).strip()
        metric_name = match.group(5).strip()
        
        # Extract value and remove any non-numeric characters
        value_text = match.group(7).strip()
        value_match = re.search(r'([0-9.]+)', value_text)
        if not value_match:
            continue
        
        value = float(value_match.group(1))
        if np.isnan(value):
            continue
        
        # Create a key for this metric
        metric_key = f"{task_name}_{filter_type}_{metric_name}"
        metrics[metric_key] = value
        
        # Determine the primary metric for this task
        if current_task == "gsm8k" and filter_type == "flexible-extract" and metric_name == "exact_match":
            task_results[current_task] = value
        elif task_name.startswith("ai2_arc") and "acc" in metric_name and not task_results.get(current_task):
            task_results[current_task] = value
        elif metric_name in ["acc", "accuracy"] and not task_results.get(current_task):
            task_results[current_task] = value
        elif metric_name == "exact_match" and not task_results.get(current_task):
            task_results[current_task] = value
    
    return metrics, task_results.get(current_task)

def main():
    args = parser.parse_args()
    
    tasks = args.tasks.split(',')
    
    task_results = {}
    all_metrics = {}
    
    for task in tasks:
        print(f"\n=== Evaluating task: {task} ===")
        
        result = subprocess.run(
            [
                "lm_eval", 
                "--model", "hf", 
                "--model_args", f"pretrained={args.model_path}",
                "--tasks", task,
                "--device", args.device,
                "--batch_size", str(args.batch_size),
                "--num_fewshot", str(args.num_fewshot)
            ],
            capture_output=True, text=True
        )
        
        print("Command executed:", " ".join(result.args))
        
        output = result.stdout
        print(output)
        
        metrics, task_result = parse_metrics(output)
        
        if metrics:
            all_metrics[task] = metrics
            if task_result is not None:
                task_results[task] = task_result
                print(f"Result for {task}: {task_result:.4f}")
            else:
                print(f"No primary metric found for {task}")
        else:
            print(f"No metrics found for {task}")
        
        if metrics:
            print(f"All metrics for {task}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
    
    print("\n=== SUMMARY OF RESULTS ===")
    print("{:<15} {:<10}".format("Task", "Performance"))
    print("-" * 25)
    
    for task, performance in task_results.items():
        print("{:<15} {:<10.4f}".format(task, performance))
    
    # Calculate average performance
    if task_results:
        avg_performance = np.mean(list(task_results.values()))
        print("-" * 25)
        print("{:<15} {:<10.4f}".format("Average", avg_performance))
    
    with open(f"{results_dir}/model_evaluation_results.md", "w") as f:
        f.write("# Model Evaluation Results\n\n")
        f.write(f"Model checkpoint: {args.model_path}\n")
        f.write(f"Number of few-shot examples: {args.num_fewshot}\n\n")
        f.write("| Task | Performance |\n")
        f.write("|------|------------|\n")
        
        for task, performance in task_results.items():
            f.write(f"| {task} | {performance:.4f} |\n")
        
        if task_results:
            f.write("|------|------------|\n")
            f.write(f"| Average | {avg_performance:.4f} |\n")
            
        # Write detailed metrics
        f.write("\n## Detailed Metrics\n\n")
        for task, metrics in all_metrics.items():
            f.write(f"### {task}\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for key, value in metrics.items():
                f.write(f"| {key} | {value:.4f} |\n")
            
            f.write("\n")
    
    print(f"\nResults saved to {results_dir}/model_evaluation_results_{args.model_path.split('/')[-1]}.md")

if __name__ == "__main__":
    main()
