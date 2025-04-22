# Model Evaluation 

## What this does

- Evaluates models trained with different methods (REGMIX, PCGRAD, CRISP)
- Runs 0-shot and 5-shot evaluations
- Generates comparison reports
- Shows which method performs best for each task and ablation

## Usage

1. Edit `run_evaluations.sh` to specify your methods and ablations:
   ```bash
   methods=(REGMIX PCGRAD CRISP)  # Change these to your methods
   ablations=(T10 T9)             # Change these to your ablations
   ```

2. Run evaluations:
   ```bash
   ./run_evaluations.sh
   ```

3. Analyze results:
   ```bash
   python analyze_results.py
   ```

4. View the generated report:
   ```bash
   cat evaluation_results/results_analysis/tables/summary.md
   ```

## Notes

- 5-shot evaluations truncate inputs to fit 512 token context window
- Default comparison baseline is REGMIX (can be changed in `analyze_results.py`)
- Results are saved as METHOD-ABLATION.json files
