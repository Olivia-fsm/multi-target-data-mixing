import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import h5py
import logging
import tiktoken

# Import your dataset loading functions
from benchmarks import get_hellaswag, get_logiqa, get_arc_easy, get_arc_challenge, get_piqa, get_sciq, get_gsm8k, get_kodcode, get_medqa, get_mathqa


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Benchmark task mapping - import the dataset loading functions
SUPPORTED_TASK_MAP = {
    "arc_easy": "get_arc_easy", 
    "arc_challenge": "get_arc_challenge", 
    "hellaswag": "get_hellaswag", 
    "logiqa": "get_logiqa", 
    "piqa": "get_piqa", 
    "sciq": "get_sciq",
    "gsm8k": "get_gsm8k",
    "kodcode": "get_kodcode",
    "medqa": "get_medqa",
    "mathqa": "get_mathqa",
}

class BenchmarkTextDataset(Dataset):
    """Dataset for benchmark decoded text (for embedding generation)"""

    def __init__(self, text_samples, tokenizer, max_length=512):
        self.texts = text_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def decode_tokens_to_text(tokens, tknzr=None):
    """Decode token IDs back into text"""
    if tknzr is None:
        tknzr = tiktoken.get_encoding("gpt2")
    return tknzr.decode(tokens)

def load_dataset_samples(dataset_name, seed=42, num_samples = None, chunk_size=512, max_total_tokens=15000000):
    """Load a sample of the dataset and decode tokens back to text"""
    np.random.seed(seed)
    
    # Get the dataset loader function
    if dataset_name == "hellaswag":
        data_dict = get_hellaswag()
    elif dataset_name == "logiqa":
        data_dict = get_logiqa()
    elif dataset_name == "arc_easy":
        data_dict = get_arc_easy()
    elif dataset_name == "arc_challenge":
        data_dict = get_arc_challenge()
    elif dataset_name == "piqa":
        data_dict = get_piqa()
    elif dataset_name == "sciq":
        data_dict = get_sciq()
    elif dataset_name == "kodcode": 
        data_dict = get_kodcode()
    elif dataset_name == "gsm8k":
        data_dict = get_gsm8k()
    elif dataset_name == "medqa":
        data_dict = get_medqa()
    elif dataset_name == "mathqa":
        data_dict = get_mathqa()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # logger.setLevel(logging.DEBUG) # NOTE : Uncomment for debugging

    logger.debug(f"Dataset: {dataset_name}")
    logger.debug(f"Total train data length: {len(data_dict['train'])}")
    logger.debug(f"Train lengths array length: {len(data_dict['train_len'])}")
    logger.debug(f"Sample lengths: {data_dict['train_len'][:10]}")
   
    # Extract data and lengths
    train_data = data_dict['train']
    train_lengths = data_dict['train_len']
    
    # Get random indices
    total_examples = len(train_lengths)

    
    if num_samples is not None:
        num_samples = min(num_samples, total_examples)
    else:
        num_samples = total_examples
    
    logger.debug(f"Total examples: {total_examples}")
    logger.debug(f"Number of samples to process: {num_samples}")
    
    indices = np.random.choice(total_examples, num_samples, replace=False)
    
    tknzr = tiktoken.get_encoding("gpt2")
    chunks = []
    all_tokens = data_dict['train']
    # track total tokens
    total_tokens = 0

    # Process in fixed-size chunks
    for i in range(0, len(all_tokens), chunk_size):
        if total_tokens >= max_total_tokens:
            logger.info(f"Reached token limit of {max_total_tokens} tokens. Stopping.")
            break
        chunk_tokens = all_tokens[i:i + chunk_size]
        # Only include chunks that have reasonable length
        if len(chunk_tokens) > chunk_size // 2: 
            chunk_text = decode_tokens_to_text(chunk_tokens, tknzr)
            chunks.append(chunk_text)
            total_tokens += len(chunk_tokens)
    logger.info(f"Loaded {len(chunks)} chunks containing approximately {total_tokens} tokens")
    return chunks
    

def get_embedding_model(model_name):
    """Load a pre-trained model for generating embeddings"""
    logger.info(f"Loading embedding model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print("Tokenizer vocabulary size:", vocab_size)

    return model, tokenizer

def generate_embeddings(model, data_loader, device, pooling_strategy='mean', normalize=True, max_batch_size=None):
    """Generate embeddings for the given token chunks"""
    model.to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Process in smaller batches if needed
            if max_batch_size is not None and input_ids.size(0) > max_batch_size:
                sub_embeddings = []
                for i in range(0, input_ids.size(0), max_batch_size):
                    end_idx = min(i + max_batch_size, input_ids.size(0))
                    sub_input_ids = input_ids[i:end_idx]
                    sub_attention_mask = attention_mask[i:end_idx]
                    
                    outputs = model(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
                    
                    if pooling_strategy == 'mean':
                        # Mean pooling - take average of all token embeddings
                        sub_emb = torch.sum(outputs.last_hidden_state * sub_attention_mask.unsqueeze(-1), dim=1)
                        sub_emb = sub_emb / torch.sum(sub_attention_mask, dim=1, keepdim=True)
                    elif pooling_strategy == 'cls':
                        # CLS pooling - use the embedding of the CLS token
                        sub_emb = outputs.last_hidden_state[:, 0, :]

                    if normalize:
                        sub_emb = sub_emb / sub_emb.norm(dim=1, keepdim=True)

                    sub_embeddings.append(sub_emb.cpu())
                
                embeddings = torch.cat(sub_embeddings, dim=0)

            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if pooling_strategy == 'mean':
                    embeddings = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
                    embeddings = embeddings / torch.sum(attention_mask, dim=1, keepdim=True)
                elif pooling_strategy == 'cls': # we don't use this
                    embeddings = outputs.last_hidden_state[:, 0, :]

                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def save_embeddings(embeddings, output_dir, benchmark_name, split):
    """Save embeddings to an HDF5 file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{benchmark_name}_{split}_embeddings.h5")
    
    logger.info(f"Saving embeddings to {output_file}")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
    
    logger.info(f"Saved {len(embeddings)} embeddings with shape {embeddings.shape}")

def main():
    

    parser = argparse.ArgumentParser(description="Generate embeddings for benchmark datasets")
    parser.add_argument("--benchmarks", nargs="+", default=list(SUPPORTED_TASK_MAP.keys()),
                        help="Benchmark datasets to process (default: all)")
    parser.add_argument("--data_module_path", type=str, default="benchmarks",
                        help="Import path for the data module containing benchmark loading functions")
    parser.add_argument("--output_dir", type=str, default="dataset_embeddings/",
                        help="Directory to save the embeddings")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model to use for generating embeddings")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding generation")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process (default: all)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Data splits to process (default: train and val)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"],
                        help="Pooling strategy for embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation")
    
    ### THIS SHOULD BE the same as the context length of the model  ######
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length for each example (longer examples will be truncated)")
    ####################################                                   

    parser.add_argument("--max_batch_tokens", type=int, default=8192,
                        help="Maximum tokens in a batch (for memory efficiency)")
    
    args = parser.parse_args()
    logger.debug(f"Benchmarks to process: {args.benchmarks}")
    # Calculate max_batch_size based on max_batch_tokens and max_length
    max_batch_size = max(1, args.max_batch_tokens // args.max_length)
    logger.info(f"Using max_batch_size of {max_batch_size} for processing")
    
    model, tokenizer = get_embedding_model(args.model_name)
    
    for benchmark in args.benchmarks:
        if benchmark not in SUPPORTED_TASK_MAP:
            logger.warning(f"Benchmark {benchmark} not in supported list, skipping")
            continue
            
        for split in args.splits:
            logger.info(f"Processing benchmark: {benchmark}, split: {split}")
            
            text_samples = load_dataset_samples(
                dataset_name=benchmark,
                seed=42
            )    

            dataset = BenchmarkTextDataset(
                text_samples=text_samples,
                tokenizer=tokenizer,
                max_length=args.max_length
            )

            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=50,
                pin_memory=True
            )

            embeddings = generate_embeddings(
                model=model, 
                data_loader=dataloader, 
                device=args.device, 
                pooling_strategy=args.pooling,
                max_batch_size=max_batch_size
            )
            
            save_embeddings(
                embeddings=embeddings, 
                output_dir=args.output_dir, 
                benchmark_name=benchmark, 
                split=split
            )
            
            logger.info(f"Completed processing {benchmark}, {split}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()