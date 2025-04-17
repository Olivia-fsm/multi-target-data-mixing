#!/usr/bin/env bash
set -euo pipefail

methods=(REGMIX PCGRAD CRISP)
ablations=(T10 T9)

base=<> # <- folder path of evaluation_model.py here
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
    suffix=${dir##*/}             # e.g. REGMIX-T10-dw[0]-tw[0]-scheduler[cosine]
    model_path="$base/$suffix/checkpoint-20000"

    python evaluate_model.py --model_path "$model_path" 

  done
done
