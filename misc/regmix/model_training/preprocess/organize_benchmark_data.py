#!/usr/bin/env python3
"""
Script to process benchmark datasets from Hugging Face, extracting TRAIN splits.
Saved in the validation directory as this is used for the evaluation of the 1M models.

Supported datasets:
- SciQ
- PIQA
- LogiQA
- Kodcode
- HellaSwag
- GSM8K
- ARC-Easy
- ARC-Challenge
- MathQA
- MedQA

Run this:
python organize_train_to_valid.py \
  --output ./sail/slimpajama-formatted \
  --datasets sciq,piqa,logiqa,kodcode,hellaswag,gsm8k,arc_easy,arc_challenge,humaneval,mathqa,medqa
"""

import os
import io
import json
import argparse
import random
import zstandard as zstd
from tqdm import tqdm
from pathlib import Path
import glob
import datasets
from datasets import load_dataset
from datasets import  disable_caching

import os

def read_zst_file(filepath):
    """Read a zstandard compressed JSONL file and yield JSON objects"""
    with open(filepath, 'rb') as f_in:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f_in) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in {filepath}")
                    continue

def process_sciq(output_dir):
    """
    Process SciQ dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading SciQ TRAIN data from Hugging Face...")
    try:
        dataset = load_dataset("sciq", split="train")
        
        print("Formatting SciQ TRAIN data for language modeling...")
        formatted_examples = []
        
        for i, example in enumerate(dataset):
            question = example['question']
            support = example.get('support', '')
            correct_answer = example['correct_answer']
            
            formatted_text = f"{support}{question}{correct_answer}"

            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "sciq",
                    "split": "valid",
                    "id": f"sciq-{i}"
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000  
        save_examples_to_files(formatted_examples, valid_dir, "sciq", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing SciQ dataset: {e}")
        return False

def process_piqa(output_dir):
    """
    Process PIQA dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading PIQA TRAIN data from Hugging Face...")
    try:
        dataset = load_dataset("piqa", split="train")
        
        print("Formatting PIQA TRAIN data for language modeling...")
        formatted_examples = []
        
        for i, example in enumerate(dataset):
            goal = example['goal']
            sol1 = example['sol1'] 
            sol2 = example['sol2']
            label = example['label']
            
            correct_sol = sol1 if label == 0 else sol2
            
            # Simplified formatting matching the second code
            formatted_text = f"{goal} {correct_sol}"
            
            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "piqa",
                    "split": "valid",
                    "id": f"piqa-{i}"
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, "piqa", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing PIQA dataset: {e}")
        return False
    
def process_logiqa(output_dir):
    """
    Process LogiQA dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading LogiQA TRAIN data from Hugging Face...")
    try:
        dataset = load_dataset("lucasmccabe/logiqa", split="train", trust_remote_code=True)
        
        print("Formatting LogiQA TRAIN data for language modeling...")
        formatted_examples = []
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"LogiQA sample data keys: {list(sample.keys())}")
        
        for i, example in enumerate(dataset):
            if 'context' in example and 'query' in example and 'options' in example and 'correct_option' in example:
                context = example['context']
                question = example['query']
                
                if isinstance(example['options'], list):
                    options = example['options']
                else:
                    # Handle case where options might be a string or other format
                    options = [example['options']]
                
                if 'correct_option' in example and isinstance(example['correct_option'], int):
                    correct_index = example['correct_option']
                    if 0 <= correct_index < len(options):
                        correct_answer = options[correct_index]
                    else:
                        print(f"Warning: Invalid correct_option index {correct_index} for example {i}")
                        correct_answer = "Unknown"
                else:
                    correct_answer = "Unknown"
                
                formatted_text = f"{context} {question} {correct_answer}"

                lm_example = {
                    "text": formatted_text,
                    "meta": {
                        "source": "logiqa",
                        "split": "valid",
                        "id": f"logiqa-{i}"
                    }
                }
                
                formatted_examples.append(lm_example)
            elif 'text' in example and 'label' in example:
                context_parts = example['text'].split("Context: ")
                if len(context_parts) > 1:
                    context = context_parts[1].split("\n")[0]
                else:
                    context = example['text']
                
                question_parts = example['text'].split("Query: ")
                if len(question_parts) > 1:
                    question = question_parts[1].split("\n")[0]
                else:
                    question = "Question not found"
                
                options = []
                correct_answer = None
                
                for j in range(1, 5):  # A, B, C, D choices
                    option_marker = f"{j}."
                    if option_marker in example['text']:
                        parts = example['text'].split(option_marker)
                        if len(parts) > j:
                            option_text = parts[j].split("\n")[0].strip()
                            options.append(option_text)
                
                if options and 'label' in example and 0 <= example['label'] < len(options):
                    correct_answer = options[example['label']]
                else:
                    print(f"Warning: Could not determine correct answer for example {i}")
                    correct_answer = "Unknown"
                
                formatted_text = f"Context: {context}\nQuestion: {question}\nAnswer: {correct_answer}"
                
                lm_example = {
                    "text": formatted_text,
                    "meta": {
                        "source": "logiqa",
                        "split": "valid",
                        "id": f"logiqa-{i}"
                    }
                }
                
                formatted_examples.append(lm_example)
            else:
                print(f"Warning: Unrecognized data format for LogiQA example {i}")
                continue
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, "logiqa", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing LogiQA dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def process_hellaswag(output_dir):
    """
    Process HellaSwag dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading HellaSwag TRAIN data from Hugging Face...")
    try:
        
        dataset = load_dataset("Rowan/hellaswag", split="train")
        
        print("Formatting HellaSwag TRAIN data for language modeling...")
        formatted_examples = []
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"HellaSwag sample data keys: {list(sample.keys())}")
        
        for i, example in enumerate(dataset):
            ctx = example['ctx']
            
            if isinstance(example['endings'], list):
                endings = example['endings']
            else:
                endings = []
                endings_dict = example['endings']
                for j in range(len(endings_dict)):
                    if str(j) in endings_dict:
                        endings.append(endings_dict[str(j)])
                    else:
                        break
            
            label_idx = None
            if isinstance(example['label'], int):
                label_idx = example['label']
            elif isinstance(example['label'], str) and example['label'].isdigit():
                label_idx = int(example['label'])
            else:
                print(f"Warning: Unexpected label format for HellaSwag example {i}: {example['label']}")
                label_idx = 0  # Default to first option
            
            # Ensure label_idx is valid
            if label_idx is not None and 0 <= label_idx < len(endings):
                correct_ending = endings[label_idx]
            else:
                print(f"Warning: Invalid label index {label_idx} for example {i}")
                correct_ending = endings[0] if endings else "No ending available"
            
            formatted_text = f"{ctx} {correct_ending}"

            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "hellaswag",
                    "split": "valid",
                    "id": f"hellaswag-{i}"
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, "hellaswag", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing HellaSwag dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_gsm8k(output_dir):
    """
    Process GSM8K dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading GSM8K TRAIN data from Hugging Face...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")

        print("Formatting GSM8K TRAIN data for language modeling...")
        formatted_examples = []
        
        for i, example in enumerate(dataset):
            question = example['question']
            answer = example['answer']
            
            formatted_text = f"{question}\n{answer}"
            
            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "gsm8k",
                    "split": "valid",
                    "id": f"gsm8k-{i}"
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, "gsm8k", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing GSM8K dataset: {e}")
        return False

def process_arc(output_dir, arc_type="easy"):
    """
    Process ARC dataset (Easy or Challenge) from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
        arc_type: "easy" or "challenge"
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    print(f"Loading ARC-{arc_type.capitalize()} TRAIN data from Hugging Face...")
    try:
 
        config_name = 'ARC-Challenge' if arc_type == "challenge" else "ARC-Easy"
        disable_caching()
        dataset = load_dataset("allenai/ai2_arc", config_name, split="train", cache_dir="/mloscratch/homes/glarou/hf_datasets_cache", trust_remote_code=True)
        
        print(f"Formatting ARC-{arc_type.capitalize()} TRAIN data for language modeling...")
        formatted_examples = []
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"ARC-{arc_type} sample data keys: {list(sample.keys())}")
        
        for i, example in enumerate(dataset):
            question = example['question']
            choices = example['choices']
            answer_key = example['answerKey']
            
            choices_text = []
            correct_answer = None
            
            if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
                for j, choice in enumerate(choices['text']):
                    label = choices['label'][j]
                    choices_text.append(f"{label}. {choice}")
                    if label == answer_key:
                        correct_answer = choice
            elif isinstance(choices, list):
                for choice in choices:
                    if 'text' in choice and 'label' in choice:
                        choices_text.append(f"{choice['label']}. {choice['text']}")
                        if choice['label'] == answer_key:
                            correct_answer = choice['text']
            
            if correct_answer is None:
                print(f"Warning: No correct answer found for ARC example {i}.")
                if len(choices_text) > 0:
                    correct_answer = choices_text[0].split('. ', 1)[1] if '. ' in choices_text[0] else choices_text[0]
                else:
                    continue  # Skip this example
            
            formatted_text = f"{question} {correct_answer}"

            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": f"arc_{arc_type}",
                    "split": "valid",
                    "id": example.get('id', f"arc_{arc_type}-{i}")
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, f"arc_{arc_type}", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing ARC-{arc_type.capitalize()} dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_kodcode(output_dir):
    """
    Process KodCode dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading KodCode TRAIN data from Hugging Face...")
    try:
        dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", trust_remote_code=True, split="train")
        
        split_dataset = dataset.train_test_split(test_size=0.1, seed=2357, shuffle=True)
        train_dataset = split_dataset["train"] # Use TRAIN part
        
        print("Formatting KodCode TRAIN data for language modeling...")
        formatted_examples = []
        
        for i, example in enumerate(train_dataset):
            question = example['question']
            solution = example['solution']
            
            formatted_text = f"{question}\n{solution}"
            
            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "kodcode",
                    "split": "valid",
                    "id": example.get('question_id', f"kodcode-{i}")
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 300000
        save_examples_to_files(formatted_examples, valid_dir, "kodcode", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing KodCode dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_mathqa(output_dir):
    """
    Process MathQA dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading MathQA TRAIN data from Hugging Face...")
    try:
        dataset = load_dataset("allenai/math_qa", split="train", trust_remote_code=True)
        
        print("Formatting MathQA TRAIN data for language modeling...")
        formatted_examples = []
        
        for i, example in enumerate(dataset):
            question = example['Problem']
            choices = {d.strip()[0]: d.split(")")[-1].strip() for d in example['options'].split(",")}
            answer = choices.get(example['correct'])
            
            formatted_text = f"{question} {answer}"

            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "mathqa",
                    "split": "valid",
                    "id": f"mathqa-{i}"
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, "mathqa", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing MathQA dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_medqa(output_dir):
    """
    Process MEDQA dataset from Hugging Face, TRAIN split
    
    Args:
        output_dir: Path to save the output files
    """
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    print("Loading MEDQA TRAIN data from Hugging Face...")
    try:

        dataset = load_dataset("bigbio/med_qa", trust_remote_code=True, split="train")
        
        print("Formatting MEDQA TRAIN data for language modeling...")
        formatted_examples = []
        
        for i, example in enumerate(dataset):
            question = example['question']
            answer = example['answer']
            
            formatted_text = f"{question} {answer}"

            lm_example = {
                "text": formatted_text,
                "meta": {
                    "source": "medqa",
                    "split": "valid",
                    "id": f"medqa-{i}"
                }
            }
            
            formatted_examples.append(lm_example)
        
        examples_per_file = 100000
        save_examples_to_files(formatted_examples, valid_dir, "medqa", examples_per_file)
        
        return True
    except Exception as e:
        print(f"Error processing MEDQA dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def save_examples_to_files(examples, output_dir, dataset_name, examples_per_file):
    """
    Save examples to multiple files
    
    Args:
        examples: List of examples to save
        output_dir: Directory to save files
        dataset_name: Name of the dataset
        examples_per_file: Number of examples per file
    """
    # Shuffle examples for even distribution
    random.shuffle(examples)
    
    # Calculate number of files needed
    num_files = max(1, len(examples) // examples_per_file + (1 if len(examples) % examples_per_file > 0 else 0))
    
    # Write to files
    for i in range(num_files):
        start_idx = i * examples_per_file
        end_idx = min((i + 1) * examples_per_file, len(examples))
        
        outfile = os.path.join(output_dir, f"{dataset_name}-{i}.jsonl")
        with open(outfile, 'w', encoding='utf-8') as f:
            for example in examples[start_idx:end_idx]:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {end_idx - start_idx} {dataset_name} examples to {outfile}")
    
    print(f"Processed {len(examples)} {dataset_name} examples across {num_files} files")

def process_dataset(dataset_name, output_dir):
    """
    Process a specific dataset
    
    Args:
        dataset_name: Name of the dataset to process
        output_dir: Path to save the output files
    """
    print(f"\n{'='*40}")
    print(f"Processing {dataset_name.upper()} dataset - TRAIN split")
    print(f"{'='*40}")
    
    if dataset_name == "sciq":
        return process_sciq(output_dir)
    elif dataset_name == "piqa":
        return process_piqa(output_dir)
    elif dataset_name == "logiqa":
        return process_logiqa(output_dir)
    elif dataset_name == "hellaswag":
        return process_hellaswag(output_dir)
    elif dataset_name == "gsm8k":
        return process_gsm8k(output_dir)
    elif dataset_name == "arc_easy":
        return process_arc(output_dir, "easy")
    elif dataset_name == "arc_challenge":
        return process_arc(output_dir, "challenge")
    elif dataset_name == "kodcode":
        return process_kodcode(output_dir)
    elif dataset_name == "medqa":
        return process_medqa(output_dir)
    elif dataset_name == "mathqa":
        return process_mathqa(output_dir)
    else:
        print(f"Unknown dataset: {dataset_name}")
        return False

def create_validation_dir(output_dir, datasets=None):
    """
    Create a validation directory with multiple datasets
    
    Args:
        output_dir: Path to save the combined validation files
        datasets: List of datasets to include
    """
    print("Creating validation directory using TRAIN data...")
    
    # Create validation directory
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    
    # Process each dataset
    dataset_success = {}
    if datasets:
        for dataset in datasets:
            dataset_success[dataset] = process_dataset(dataset, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("Validation Directory Creation Summary (TRAIN data)")
    print("="*60)
    
    if datasets:
        for dataset in datasets:
            print(f"{dataset.upper()}: {'Success' if dataset_success.get(dataset, False) else 'Failed'}")
    
    # Count files by type
    for dataset in (datasets or []):
        dataset_files = len(glob.glob(os.path.join(valid_dir, f"{dataset}-*.jsonl")))
        if dataset_files > 0:
            print(f"{dataset.upper()} files: {dataset_files}")
    
    print(f"\nValidation directory created at: {valid_dir}")
    
    # Return overall success
    all_success = True
    if datasets:
        all_success = all(dataset_success.values())
    
    return all_success


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process benchmark datasets from Hugging Face, extracting TRAIN splits but saving in valid directory")
    parser.add_argument("--output", required=True, 
                       help="Path to save the processed files")
    parser.add_argument("--datasets", 
                       help="Comma-separated list of datasets to include (sciq,piqa,logiqa,kodcode,hellaswag,gsm8k,arc_easy,arc_challenge,humaneval,mathqa,medqa)")
    
    args = parser.parse_args()
    
    # Parse datasets
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    
    # If no datasets specified, include all
    if not datasets:
        datasets = [ "arc_easy", "arc_challenge","sciq", "piqa", "logiqa", "hellaswag", "gsm8k", "kodcode", "humaneval", "mathqa", "medqa"]

    create_validation_dir(args.output, datasets)
