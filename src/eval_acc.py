import os
import subprocess
import numpy as np
import itertools
from tqdm import tqdm
import argparse

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--model_dir', default='/scratch/homes/sfan/multi_doge/exp/MAP-8tasks-684M-dw[100]-tw[100]-scheduler[decay]', type=str)
args_parser.add_argument('--output_dir', default='/scratch/homes/sfan/multi_doge/eval_results', type=str)
args_parser.add_argument('--save_path', default='/scratch/homes/sfan/multi_doge/eval_results/map.pkl', type=str)
args_parser.add_argument('--task_list', default="ai2_arc,hellaswag,logiqa,piqa,sciq,gsm8k,humaneval", type=str)

def eval_model(ckpt_path, 
              task_list):
    acc_dict = {}
    for task in task_list:
        print(f"Evaluating task: {task}")
        print(ckpt_path)
        result = subprocess.run(
                    [
                        "lm_eval", 
                        "--model", "hf", 
                        "--model_args", f"pretrained={ckpt_path}",
                        "--tasks", task,
                        "--device", "cuda:0",
                        "--batch_size", "64"
                    ],
                    capture_output=True, text=True
        )
        print("Command executed:", " ".join(result.args))

        output = result.stdout
        print(output)
        for line in output.splitlines():
            if "acc" in line:
                accuracy = float(line.split('|')[7].strip())  # Extract accuracy value
                acc_dict[task] = accuracy  # Save the accuracy for the task
        print(f"Standalone results for {task}: {acc_dict[task]}")
    return acc_dict

def main():
    args = args_parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_dir = args.model_dir
    
    all_acc = {}
    for ckpt in tqdm(os.listdir(model_dir)):
        ckpt_path = os.path.join(model_dir, ckpt)
        ckpt_acc = eval_model(ckpt_path, args.task_list.split(","))
        all_acc[ckpt] = ckpt_acc
    
    with open(args.save_path, "wb") as trg:
        pickle.dump(all_acc, trg)

if __name__ == "__main__":
    main()