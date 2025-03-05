import os
import subprocess
import numpy as np
from scipy.stats import ttest_rel
import itertools

# Directories
base_dir = "chkpt/base"
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--model_dir', default='/mloraw1/sfan/doge/config/doge_82.json', type=str)
args_parser.add_argument('--output_dir', default='multi-target-reweight_684M', type=str)
args_parser.add_argument('--task_list', default="ai2arc,hellaswag,logiqa,piqa,sciq", type=str)

def eval_task(model_ckpt_path, task_list):
    for task in tasks:
        print(f"Evaluating standalone task: {task}")
        model_dir = f"{base_dir}/{task}"
        print(model_dir)
        result = subprocess.run(
                    [
                        "lm_eval", 
                        "--model", "hf", 
                        "--model_args", f"pretrained={model_dir}",
                        "--tasks", task,
                        "--device", "cuda:0",
                        "--batch_size", "32"
                    ],
                    capture_output=True, text=True
        )
        print("Command executed:", " ".join(result.args))

        output = result.stdout
        print(output)
        for line in output.splitlines():
            if "acc" in line:
                accuracy = float(line.split('|')[7].strip())  # Extract accuracy value
                standalone_accuracies[task] = accuracy  # Save the accuracy for the task
        print(f"Standalone results for {task}: {standalone_accuracies[task]}")
    return

def main(args):
    args = args_parser.parse_args()
    
    
    methods = {}
    number_of_target_domains = 2
    for dataset_combo in itertools.combinations(tasks, number_of_target_domains):
            methods[f"{'_'.join(dataset_combo)}_mixture"] = list(dataset_combo)
            methods[f"{'_'.join(dataset_combo)}_interpolation"] = list(dataset_combo)

    standalone_accuracies = {}

    # Step 1: Evaluate standalone accuracies for each task
    if standalone_accuracies == {}:

        for task in tasks:
            print(f"Evaluating standalone task: {task}")
            model_dir = f"{base_dir}/{task}"
            print(model_dir)
            result = subprocess.run(
                        [
                            "lm_eval", 
                            "--model", "hf", 
                            "--model_args", f"pretrained={model_dir}",
                            "--tasks", task,
                            "--device", "cuda:0",
                            "--batch_size", "32"
                        ],
                        capture_output=True, text=True
            )
            print("Command executed:", " ".join(result.args))

            output = result.stdout
            print(output)
            for line in output.splitlines():
                if "acc" in line:
                    accuracy = float(line.split('|')[7].strip())  # Extract accuracy value
                    standalone_accuracies[task] = accuracy  # Save the accuracy for the task
            print(f"Standalone results for {task}: {standalone_accuracies[task]}")

    # Step 2: Evaluate methods (mixtures/interpolations)

    method_results = {}

    if method_results == {} :
        for method_name, targeted_tasks in methods.items():
            print(f"Evaluating method: {method_name}")
            method_results[method_name] = {}
            for task in targeted_tasks:
                model_dir = f"{base_dir}/{method_name}"
                print(model_dir)
                result = subprocess.run(
                    [
                        "lm_eval", 
                        "--model", "hf", 
                        "--model_args", f"pretrained={model_dir}",
                        "--tasks", task,
                        "--device", "cuda:0",
                        "--batch_size", "32"
                    ],
                    capture_output=True, text=True
                )
                output = result.stdout
                print(output)
                for line in output.splitlines():
                    if "acc" in line:
                        accuracy = float(line.split('|')[7].strip())  # Extract accuracy value
                        method_results[method_name][task] = accuracy  # Save the accuracy for the task
            print(f"Method results for {method_name}: {method_results[method_name]}")

    print(f"METHOD RESULTS : {method_results}")

    # Step 3: Compare method accuracies to baseline
    evaluation_results = []

    for method_name, method_accuracies in method_results.items():
        targeted_tasks = methods[method_name]
        
        # Calculate baseline average accuracy for targeted tasks
        baseline_avg = np.mean([standalone_accuracies[task] for task in targeted_tasks])
        
        # Calculate method's average accuracy for targeted tasks
        method_avg = np.mean(list(method_accuracies.values()))
        
        # Compute the difference
        delta = method_avg - baseline_avg
        
        # Store results
        evaluation_results.append({
            'method': method_name,
            'targeted_tasks': targeted_tasks,
            'acc_1':standalone_accuracies[targeted_tasks[0]],
            'acc_2':standalone_accuracies[targeted_tasks[1]],
            'baseline_avg': baseline_avg,
            'acc_1_method':list(method_accuracies.values())[0],
            'acc_2_method':list(method_accuracies.values())[1],
            'method_avg': method_avg,
            'delta': delta
        })
    # Sort by improvement (highest improvement first)
    evaluation_results.sort(key=lambda x: x['delta'], reverse=True)
    # Summarize results
    print("\nMethod Evaluation Results:")
    print("{:<25} {:<20} {:<25} {:<25} {:<15} {:<25} {:<25} {:<15} {:<10}".format(
        "Method", "Targeted Tasks", "Original Acc. of Task 1", "Original Acc. of Task 2", "Baseline Avg", "Attained Acc. of Task 1", "Attained Acc. of Task 2", "Method Avg", "Improvement"
    ))

    for result in evaluation_results:
        print("{:<25} {:<20} {:<25.3f} {:<25.3f} {:<15.3f} {:<25.3f} {:<25.3f} {:<15.3f} {:<10.3f} ".format(
            result['method'],
            ', '.join(result['targeted_tasks']),
            result['acc_1'],
            result['acc_2'],
            result['baseline_avg'],
            result['acc_1_method'],
            result['acc_2_method'],
            result['method_avg'],
            result['delta']
        ))



    # Step 4: Compute per-task improvements and sort methods by best improvement on a single task
    per_task_improvements = []

    for method_name, method_accuracies in method_results.items():
        targeted_tasks = methods[method_name]
        
        for i, task in enumerate(targeted_tasks):
            standalone_acc = standalone_accuracies[task]
            method_acc = list(method_accuracies.values())[i]
            improvement = method_acc - standalone_acc
            
            per_task_improvements.append({
                'method': method_name,
                'task': task,
                'standalone_acc': standalone_acc,
                'method_acc': method_acc,
                'improvement': improvement
            })

    # Sort by improvement (highest improvement first)
    per_task_improvements.sort(key=lambda x: x['improvement'], reverse=True)

    # Print the sorted improvements
    print("\nMethods Ranked by Best Improvement on a Single Task:")
    print("{:<25} {:<10} {:<20} {:<20} {:<15}".format(
        "Method", "Task", "Standalone Acc", "Method Acc", "Improvement"
    ))
    for entry in per_task_improvements:
        print("{:<25} {:<10} {:<20.3f} {:<20.3f} {:<15.3f}".format(
            entry['method'],
            entry['task'],
            entry['standalone_acc'],
            entry['method_acc'],
            entry['improvement']
        ))

    # Save results to a markdown file
    with open("outputs/unbalanced_val/accuracy_eval_results.md", "w") as md_file:

        print("\nMethod Evaluation Results:", file = md_file)
        print("{:<25} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<10}".format(
            "Method", "Targeted Tasks", "Original Acc. of Task 1", "Original Acc. of Task 2", "Baseline Avg", "Attained Acc. of Task 1", "Attained Acc. of Task 2", "Method Avg", "Delta"
        ), file = md_file)
        for result in evaluation_results:
            print("{:<25} {:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<10.3f} ".format(
                result['method'],
                ', '.join(result['targeted_tasks']),
                result['acc_1'],
                result['acc_2'],
                result['baseline_avg'],
                result['acc_1_method'],
                result['acc_2_method'],
                result['method_avg'],
                result['delta']
            ), file = md_file)

        print("\nMethods Ranked by Best Improvement on a Single Task:", file=md_file)
        print("{:<25} {:<10} {:<20} {:<20} {:<15}".format(
            "Method", "Task", "Standalone Acc", "Method Acc", "Improvement"
        ), file=md_file)
        for entry in per_task_improvements:
            print("{:<25} {:<10} {:<20.3f} {:<20.3f} {:<15.3f}".format(
                entry['method'],
                entry['task'],
                entry['standalone_acc'],
                entry['method_acc'],
                entry['improvement']
            ), file=md_file)

    print("\nSaved to 'outputs/unbalanced_val/accuracy_eval_results.md'.")


if __name__ == "__main__":
    main()