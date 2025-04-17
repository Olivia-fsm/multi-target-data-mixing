"""
Python script to process pre-tokenized binary files into PackedDataset format
for efficient language model training.

Usage:
python prepare_tokenized_data.py \
  --source_path /path/to/tokenized/files/ \
  --destination_path /path/to/output/packed_dataset/ \
  --split train
"""
import glob
import os
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count, Pool
import time

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset

# Define the languages you want to process
TRAIN_LANGUAGES = ["en", "fr", "de", "es", "it","ru"]
VALID_LANGUAGES = [ "pl", "uk", "nl", "pt", "ca", "tr", "da", "ro"] 
ALL_LANGUAGES = TRAIN_LANGUAGES + VALID_LANGUAGES

# Define sets - using a pattern that works with multiple languages
tokenized_data_sets = {
    "train": "{" + ",".join(TRAIN_LANGUAGES) + "}/*_train.bin",
    "valid": "{" + ",".join(VALID_LANGUAGES) + "}/*_train.bin"
}

class TokenizedPackedDatasetBuilder(packed_dataset.PackedDatasetBuilder):
    """
    Extended PackedDatasetBuilder for pre-tokenized data that customizes
    the filename format without reimplementing the core functionality.
    """
    def __init__(self, outdir, prefix, chunk_size, language, file_index=0, **kwargs):
        # Create a language-specific prefix
        self._language = language
        self._file_index = file_index
        # Use the parent class's implementation but with our custom prefix
        modified_prefix = f"{prefix}_wiki40b_{language}-{file_index:05d}"
        super().__init__(outdir=outdir, prefix=modified_prefix, chunk_size=chunk_size, **kwargs)

def extract_language_from_path(filepath):
    """Extract language code from a filepath"""
    # Try to find language code in the path
    for lang in ALL_LANGUAGES:
        if f"/{lang}/" in filepath or filepath.startswith(f"{lang}/"):
            return lang
    
    # If not found via path structure, try filename
    filename = os.path.basename(filepath)
    for lang in ALL_LANGUAGES:
        if filename.startswith(lang + "_") or f"_{lang}_" in filename:
            return lang
    
    # If still not found, extract from directory name
    parent_dir = os.path.basename(os.path.dirname(filepath))
    if parent_dir in ALL_LANGUAGES:
        return parent_dir
    
    # Default fallback
    print(f"Warning: Could not detect language for {filepath}, using 'unknown'")
    return "unknown"

def prepare_tokenized(
    source_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str = "train",
    filenames_subset: list = None,
    process_id: int = 0,
    vocab_size: int = 256158,  # Default XLM tokenizer vocab size
    sep_token: int = 2         # Default </s> token ID in XLM tokenizer
) -> None:
    """
    Process pre-tokenized binary files and convert them to the packed dataset format.
    
    Args:
        source_path: Base path containing tokenized files
        destination_path: Output directory for packed dataset files
        chunk_size: Size of each chunk in the packed dataset
        split: Data split ('train' or 'valid')
        filenames_subset: Specific files to process (if None, all files are processed)
        process_id: ID for parallel processing
        vocab_size: Size of the vocabulary (for dtype selection)
        sep_token: Separator token ID (usually BOS or EOS token)
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset
    
    if not filenames:
        raise RuntimeError(
            f"No files matching {tokenized_data_sets[split]} found at {source_path}. \n"
            "Make sure you have the pre-tokenized files available."
        )
    
    # Import struct for binary packing (needed by the builder)
    import struct
    
    # Process each file
    for filepath in tqdm(filenames, desc=f"Process {process_id}"):
        try:
            # Extract language from the filepath
            language = extract_language_from_path(filepath)
            
            # Create builder with the language-specific naming
            builder = TokenizedPackedDatasetBuilder(
                outdir=destination_path,
                prefix=split,  # 'train' or 'valid'
                chunk_size=chunk_size,
                language=language,
                file_index=process_id * 1000,  # Ensure uniqueness across processes
                sep_token=sep_token,
                dtype="auto",
                vocab_size=vocab_size,
            )
            
            # Load the pre-tokenized binary data
            file_size = os.path.getsize(filepath)
            
            if vocab_size <= 65535:
                token_dtype = np.uint16
            else:
                token_dtype = np.int32
                
            # Calculate number of tokens based on file size and dtype
            token_size = np.dtype(token_dtype).itemsize
            num_tokens = file_size // token_size
            
            # Load the pre-tokenized binary data
            token_ids = np.memmap(filepath, dtype=token_dtype, mode='r', shape=(num_tokens,))
            
            # Process in chunks to avoid loading the entire file into memory
            # Choose a reasonable chunk size (e.g., 10 times the packed chunk size)
            processing_chunk_size = chunk_size * 10
            
            for i in range(0, len(token_ids), processing_chunk_size):
                # Get a chunk of token IDs
                end_idx = min(i + processing_chunk_size, len(token_ids))
                chunk = np.array(token_ids[i:end_idx], dtype=builder.dtype)
                
                # Add the token IDs to the builder
                builder.add_array(chunk)
            
            # Write any remaining tokens
            if builder._idx > 0:
                builder.write_reminder()
                
            print(f"Successfully processed {filepath} ({language}), created {len(builder.filenames)} output files")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
def process_file(args):
    """Helper function for parallel processing"""
    source_path, destination_path, chunk_size, split, filename, index, vocab_size, sep_token = args
    prepare_tokenized(
        source_path, 
        destination_path, 
        chunk_size, 
        split, 
        [filename], 
        index,
        vocab_size,
        sep_token
    )

def prepare(
    source_path: Path = Path("path/to/tokenized/data"),
    destination_path: Path = Path("data/packed_tokenized"),
    chunk_size: int = 2049 * 256 ,
    split: str = "train",
    percentage: float = 1.0,
    vocab_size: int = 256158,  # Adjust based on your tokenizer's vocab size
    sep_token: int = 2,       # The </s> token ID in XLM tokenizer is typically 2
    n_processes: int = None,
) -> None:
    """
    Main function to prepare pre-tokenized binary files into packed dataset format.
    
    Args:
        source_path: Path containing tokenized binary files
        destination_path: Output path for packed dataset
        chunk_size: Size of each chunk in the packed dataset
        split: Data split ('train', 'valid', or 'all')
        percentage: Percentage of files to process (1.0 = all files)
        vocab_size: Size of the vocabulary
        sep_token: Separator token ID
        n_processes: Number of parallel processes to use
    """
    # Process both train and valid splits if requested
    splits_to_process = ["train", "valid"] if split == "all" else [split]
    
    for current_split in splits_to_process:
        pattern = os.path.join(str(source_path), tokenized_data_sets[current_split])
        print(f"Looking for files matching pattern: {pattern}")
        filenames = glob.glob(pattern, recursive=True)
        
        # If the glob pattern with braces doesn't work in your environment
        # you may need this alternative approach:
        if not filenames:
            # Try individual patterns for each language
            filenames = []
            languages_for_split = TRAIN_LANGUAGES if current_split == "train" else VALID_LANGUAGES
            for lang in languages_for_split:
                lang_pattern = os.path.join(str(source_path), f"{lang}/*_train.bin")
                lang_files = glob.glob(lang_pattern, recursive=True)
                filenames.extend(lang_files)
        
        if not filenames:
            print(f"Warning: No files found matching pattern {pattern}")
            continue
        
        # Apply percentage if requested
        if percentage < 1.0:
            num_files = max(1, int(len(filenames) * percentage))
            filenames = filenames[:num_files]
        
        print(f"Processing {len(filenames)} files for {current_split} split")
        
        tasks = []
        for i, filename in enumerate(filenames):
            tasks.append(
                (source_path, destination_path, chunk_size, current_split, filename, i, vocab_size, sep_token)
            )
        
        if n_processes is None:
            n_processes = min(cpu_count(), len(filenames))
        else:
            n_processes = min(n_processes, len(filenames))
        
        print(f"Using {n_processes} processes")
        
        start_time = time.time()
        
        with Pool(processes=n_processes) as pool:
            pool.map(process_file, tasks)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for {current_split} split: {elapsed_time:.2f} seconds")
        print(f"Output files saved to {destination_path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
