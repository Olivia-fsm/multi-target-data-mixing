#!/usr/bin/env bash
set -euo pipefail

methods=(REGMIX PCGRAD CRISP)
ablations=(T10 T9)

base=<>
mkdir -p results/raw

for ablation in "${ablations[@]}"; do
  echo
  echo "=== Ablation: $ablation ==="
  printf "%-10s  %s\n" Method  Metric
  for method in "${methods[@]}"; do
    # find the one matching dir, strip path to get suffix
    dir=$(ls -d "$base"/${method}-${ablation}-* 2>/dev/null)
    if [[ -z "$dir" ]]; then
      echo "No experiment dir for ${method}-${ablation}" >&2
      continue
    fi
    suffix=${dir##*/}             
    model_path="$base/$suffix/checkpoint-20000"

    python evaluate_model.py --model_path "$model_path" --num_fewshot 0 --output_dir "evaluation_results/0_shot"
    python evaluate_model.py --model_path "$model_path" --num_fewshot 5 --output_dir "evaluation_results/5_shot"

  done
done
