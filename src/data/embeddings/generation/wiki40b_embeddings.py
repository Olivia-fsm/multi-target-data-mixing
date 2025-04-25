import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import XLMTokenizer, XLMModel
from torch.utils.data import DataLoader, Dataset
import h5py

import urllib3
urllib3.util.timeout.Timeout.DEFAULT_TIMEOUT = 30  # or higher

# Define constants
WIKI_40B_PATH = "/mloscratch/homes/sfan/multilingual-wiki-data/"
OUTPUT_PATH = "dataset_embeddings/wiki40b_embeddings/"  # Where to save embeddings
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512  # Maximum sequence length for the model
MAX_TOKENS = 20_000_000  # Maximum number of tokens to process per language
LANGUAGES = [ "en", "fr", "de", "es", "it", "ru", "pl", "uk", "nl", "pt", "ca", "tr", "da","ro"]
#
# Create a custom dataset to handle the memmap data
class Wiki40BDataset(Dataset):
    def __init__(self, data, max_seq_length=512):
        self.data = data
        self.max_seq_length = max_seq_length
        
        # Determine total number of sequences
        self.total_seqs = len(self.data) // self.max_seq_length
        if len(self.data) % self.max_seq_length > 0:
            self.total_seqs += 1
    
    def __len__(self):
        return self.total_seqs
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_length
        end_idx = min(start_idx + self.max_seq_length, len(self.data))
        
        # Get sequence and create attention mask
        sequence = self.data[start_idx:end_idx].clone()
        attention_mask = torch.ones_like(sequence)
        
        # If sequence is shorter than max_seq_length, pad it
        if end_idx - start_idx < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - (end_idx - start_idx), dtype=sequence.dtype)
            sequence = torch.cat([sequence, padding])
            attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)])
            
        return {
            'input_ids': sequence,
            'attention_mask': attention_mask
        }

def load_wiki40b_data(subset='en', max_tokens=MAX_TOKENS):
    """Load Wiki40B training data for a specific language, limiting to max_tokens."""
    subset_path = os.path.join(WIKI_40B_PATH, subset)
    train_path = os.path.join(subset_path, f"{subset}_train.bin")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    # Load data as memmap first
    full_train_data = np.memmap(train_path, dtype=np.int32, mode='r')
    
    # Limit the number of tokens
    token_count = min(len(full_train_data), max_tokens)
    train_data = full_train_data[:token_count]
    
    # Convert to torch tensor
    train_data = torch.tensor(np.array(train_data, dtype=np.int32))
    
    print(f'Loaded {subset} training data: {len(train_data)} tokens (limited from {len(full_train_data)})')
    return train_data

def create_embeddings(model, dataset, device, output_file, chunk_size=10000):
    """Generate embeddings for all sequences in the dataset and save them."""
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create HDF5 file to store embeddings
    with h5py.File(output_file, 'w') as f:
        # Create dataset with unknown size initially, will resize as we add data
        embeddings_dataset = f.create_dataset(
            'embeddings', 
            shape=(0, model.config.hidden_size), 
            maxshape=(None, model.config.hidden_size),
            chunks=(chunk_size, model.config.hidden_size),
            dtype='f'
        )
        
        # Position counter for writing to HDF5
        current_position = 0
        
        # Process batches
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use CLS token ([CLS]) embeddings from the last hidden state
                # Shape: [batch_size, hidden_size]
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Resize dataset to accommodate new embeddings
                batch_size = cls_embeddings.shape[0]
                embeddings_dataset.resize(current_position + batch_size, axis=0)
                
                # Save embeddings
                embeddings_dataset[current_position:current_position + batch_size] = cls_embeddings
                current_position += batch_size

def main():
    parser = argparse.ArgumentParser(description='Create embeddings for Wiki40B training data.')
    parser.add_argument('--languages', nargs='+', default=LANGUAGES, 
                        help='Languages to process (space-separated list)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_PATH,
                        help='Directory to save embeddings')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS,
                        help='Maximum number of tokens to process per language')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading XLM model and tokenizer...")

    tokenizer = XLMTokenizer.from_pretrained('FacebookAI/xlm-mlm-100-1280', cache_dir=None, force_download=True)
    model = XLMModel.from_pretrained('FacebookAI/xlm-mlm-100-1280', cache_dir=None, force_download=True)
    device = torch.device(args.device)
    model.to(device)
    print(f"Model loaded. Using device: {device}")
    
    # Process each language
    for lang in args.languages:
        try:
            print(f"\nProcessing language: {lang}")
            
            # Load data with token limit
            train_data = load_wiki40b_data(lang, args.max_tokens)
            
            # Create dataset
            dataset = Wiki40BDataset(train_data, max_seq_length=MAX_SEQ_LENGTH)
            print(f"Created dataset with {len(dataset)} sequences")
            
            # Create output file path
            output_file = os.path.join(args.output_dir, f"{lang}_embeddings.h5")
            
            # Generate and save embeddings
            print(f"Generating embeddings and saving to {output_file}")
            create_embeddings(model, dataset, device, output_file)
            
            print(f"Completed processing for {lang}")
            
            # Free up memory
            del train_data, dataset
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing language {lang}: {str(e)}")
    
    print("\nEmbedding generation complete!")

if __name__ == "__main__":
    main()
