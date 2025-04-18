import os
import json
import re
import pandas as pd
from collections import defaultdict

def analyze_results(directory_path, pattern=r'([A-Z]+)-([A-Z0-9]+)\.json'):
    results_by_ablation = defaultdict(dict)
    methods = set()
    tasks = set()
    
    for filename in os.listdir(directory_path):
        match = re.match(pattern, filename)
        if match and filename.endswith('.json'):
            method, ablation = match.groups()
            methods.add(method)
            
            with open(os.path.join(directory_path, filename), 'r') as f:
                data = json.load(f)
            
            task_results = {}
            for task_name, task_data in data['results'].items():
                tasks.add(task_name)
                task_results[task_name] = task_data['acc_norm']
            
            results_by_ablation[ablation][method] = {
                'average_acc_norm': data['average_acc_norm'],
                'task_results': task_results
            }
    
    all_data = []
    for ablation, method_data in results_by_ablation.items():
        for method, results in method_data.items():
            for task, accuracy in results['task_results'].items():
                all_data.append({
                    'Method': method,
                    'Ablation': ablation,
                    'Task': task,
                    'Accuracy': accuracy,
                    'Metric': 'acc_norm'
                })
            
            all_data.append({
                'Method': method,
                'Ablation': ablation,
                'Task': 'Average',
                'Accuracy': results['average_acc_norm'],
                'Metric': 'acc_norm'
            })
    
    return pd.DataFrame(all_data), list(methods), list(tasks)

def create_summary_markdown(results_df, output_path, base_method="REGMIX"):
    os.makedirs(os.path.dirname(os.path.join(output_path, 'summary.md')), exist_ok=True)
    
    with open(os.path.join(output_path, 'summary.md'), 'w') as md_file:
        md_file.write(f"# Experiment Results (Normalized Accuracy)\n\n")
        md_file.write(f"*Base method for comparisons: **{base_method}***\n\n")
        
        norm_df = results_df[results_df['Metric'] == 'acc_norm']
        methods = sorted(norm_df['Method'].unique())
        ablations = sorted(norm_df['Ablation'].unique())
        
        base_method_exists = base_method in methods
        if not base_method_exists:
            md_file.write(f"**Warning:** Base method '{base_method}' not found in results. Using absolute values only.\n\n")
        
        md_file.write("## Comparison by Ablation\n\n")
        
        for ablation in ablations:
            md_file.write(f"### Ablation: {ablation}\n\n")
            
            ablation_df = norm_df[norm_df['Ablation'] == ablation]
            pivot_df = ablation_df.pivot(index='Task', columns='Method', values='Accuracy')
            
            if base_method_exists:
                relative_df = pivot_df.copy()
                
                for method in methods:
                    if method != base_method:
                        relative_df[f"{method} vs {base_method}"] = pivot_df[method] - pivot_df[base_method]
                
                ordered_columns = [base_method]
                for method in [m for m in methods if m != base_method]:
                    ordered_columns.extend([method, f"{method} vs {base_method}"])
                
                ordered_columns = [col for col in ordered_columns if col in relative_df.columns]
                relative_df = relative_df[ordered_columns]
                
                md_file.write(relative_df.to_markdown(floatfmt=".4f"))
            else:
                md_file.write(pivot_df.to_markdown(floatfmt=".4f"))
                
            md_file.write("\n\n")
            
            if not pivot_df.empty:
                best_task_method = {}
                for task in pivot_df.index:
                    if task != 'Average':
                        best_method = pivot_df.loc[task].idxmax()
                        best_score = pivot_df.loc[task, best_method]
                        best_task_method[task] = (best_method, best_score)
                
                if best_task_method:
                    md_file.write("#### Best Methods per Task\n\n")
                    best_task_df = pd.DataFrame(
                        [(task, method, score) for task, (method, score) in best_task_method.items()],
                        columns=['Task', 'Best Method', 'Score']
                    )
                    md_file.write(best_task_df.to_markdown(floatfmt=".4f"))
                    md_file.write("\n\n")
                
                if 'Average' in pivot_df.index:
                    best_method = pivot_df.loc['Average'].idxmax()
                    best_score = pivot_df.loc['Average', best_method]
                    
                    if base_method_exists and best_method != base_method:
                        base_score = pivot_df.loc['Average', base_method]
                        improvement = best_score - base_score
                        percent_improvement = (improvement / base_score) * 100 if base_score > 0 else float('inf')
                        
                        md_file.write(f"Best method for ablation {ablation}: **{best_method}** " +
                                     f"(average: {best_score:.4f}, " +
                                     f"+{improvement:.4f} absolute / +{percent_improvement:.2f}% relative improvement over {base_method})\n\n")
                    else:
                        md_file.write(f"Best method for ablation {ablation}: **{best_method}** (average: {best_score:.4f})\n\n")
        
        md_file.write("## Comparison by Method\n\n")
        
        for method in methods:
            if method == base_method:
                continue
                
            md_file.write(f"### Method: {method}\n\n")
            
            method_df = norm_df[norm_df['Method'] == method]
            pivot_df = method_df.pivot(index='Task', columns='Ablation', values='Accuracy')
            
            md_file.write(pivot_df.to_markdown(floatfmt=".4f"))
            md_file.write("\n\n")
            
            if 'Average' in pivot_df.index:
                best_ablation = pivot_df.loc['Average'].idxmax()
                best_score = pivot_df.loc['Average', best_ablation]
                md_file.write(f"Best ablation for {method}: **{best_ablation}** (average: {best_score:.4f})\n\n")
            
            if base_method_exists:
                for ablation in ablations:
                    md_file.write(f"#### Tasks Ordered by Improvement Over {base_method} for Ablation {ablation}\n\n")
                    
                    method_results = norm_df[
                        (norm_df['Method'] == method) & 
                        (norm_df['Ablation'] == ablation) & 
                        (norm_df['Task'] != 'Average')
                    ]
                    
                    base_results = norm_df[
                        (norm_df['Method'] == base_method) & 
                        (norm_df['Ablation'] == ablation) & 
                        (norm_df['Task'] != 'Average')
                    ]
                    
                    improvements = []
                    
                    for _, method_row in method_results.iterrows():
                        task = method_row['Task']
                        method_acc = method_row['Accuracy']
                        
                        base_acc = base_results[base_results['Task'] == task]['Accuracy'].values
                        if len(base_acc) > 0:
                            improvement = method_acc - base_acc[0]
                            rel_improvement = (improvement / base_acc[0] * 100) if base_acc[0] > 0 else float('inf')
                            
                            improvements.append({
                                'Task': task,
                                'Base Acc': base_acc[0],
                                'Method Acc': method_acc,
                                'Abs Improvement': improvement,
                                'Rel Improvement (%)': rel_improvement
                            })
                    
                    if improvements:
                        improvements_df = pd.DataFrame(improvements)
                        improvements_df = improvements_df.sort_values('Abs Improvement', ascending=False)
                        
                        formatted_df = improvements_df.copy()
                        formatted_df['Abs Improvement'] = formatted_df['Abs Improvement'].map(lambda x: f"{x:.4f}")
                        formatted_df['Rel Improvement (%)'] = formatted_df['Rel Improvement (%)'].map(lambda x: f"{x:.2f}%" if x != float('inf') else "∞")
                        formatted_df['Base Acc'] = formatted_df['Base Acc'].map(lambda x: f"{x:.4f}")
                        formatted_df['Method Acc'] = formatted_df['Method Acc'].map(lambda x: f"{x:.4f}")
                        
                        md_file.write(formatted_df.to_markdown(index=False))
                        md_file.write("\n\n")
                    else:
                        md_file.write(f"No direct comparisons available with {base_method} for ablation {ablation}.\n\n")

def main():
    directory_path = 'evaluation_results'
    output_dir = 'evaluation_results/results_analysis'
    base_method = 'REGMIX'
    
    results_df, methods, tasks = analyze_results(directory_path)
    
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    create_summary_markdown(results_df, os.path.join(output_dir, 'tables'), base_method)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"\nMethods found: {', '.join(methods)}")
    print(f"Ablations found: {', '.join(results_df['Ablation'].unique())}")
    print(f"Tasks found: {', '.join(tasks)}")
    
    avg_df = results_df[(results_df['Task'] == 'Average') & (results_df['Metric'] == 'acc_norm')]
    print("\nAverage Performance Summary:")
    print(avg_df.pivot(index='Ablation', columns='Method', values='Accuracy'))
    
    print(f"\nMarkdown summary available at: {os.path.join(output_dir, 'tables/summary.md')}")
    print(f"Base method for comparisons: {base_method} 🦋")


if __name__ == "__main__":
    main()
    print("🎉 Done!")
