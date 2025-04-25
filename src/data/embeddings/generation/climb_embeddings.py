import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
import h5py
import tiktoken
import urllib3
urllib3.util.timeout.Timeout.DEFAULT_TIMEOUT = 30

# Define constants
CLIMBLAB_PATH = "/mloscratch/homes/sfan/multi_doge/src/data/datasets/climblab/"
OUTPUT_PATH = "dataset_embeddings/climblab/"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512
CHUNK_SIZE = 512
MAX_TOKENS = 20_000_000
CLUSTERS = [f"cluster_{i}" for i in range(1, 21)]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_CHUNK_SIZE = 4096

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]



def load_and_decode_climblab_data(cluster='cluster_1', max_tokens=MAX_TOKENS, chunk_size=512):
    cluster_path = os.path.join(CLIMBLAB_PATH, cluster)
    train_path = os.path.join(cluster_path, "train.bin")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    print(f"Loading GPT-2 tokenizer...")
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    
    file_size = os.path.getsize(train_path)
    print(f"File size: {file_size} bytes")
    
    try:
        print("Reading file with np.memmap using np.uint16...")
        full_train_data = np.memmap(
            train_path, dtype=np.uint16, mode='r', offset=0
        )
        
        total_tokens_available = len(full_train_data)
        total_chunks_needed = max_tokens // chunk_size
        total_chunks_available = total_tokens_available // chunk_size
        
        num_chunks = min(total_chunks_needed, total_chunks_available)
        max_chunks_to_process = num_chunks
        
        print(f"Will process {max_chunks_to_process} chunks of {chunk_size} tokens each")
        print(f"Total tokens to process: {max_chunks_to_process * chunk_size} (limit: {max_tokens})")
        
        text_chunks = []
        tokens_processed = 0
        
        for chunk_idx in tqdm(range(max_chunks_to_process), desc="Processing token chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            
            chunk_tokens = full_train_data[start_idx:end_idx].astype(np.int32).tolist()
            
            try:
                chunk_text = gpt2_tokenizer.decode(chunk_tokens)
                text_chunks.append(chunk_text)
                tokens_processed += len(chunk_tokens)
                
                if tokens_processed >= max_tokens:
                    print(f"Reached token limit of {max_tokens}. Stopping.")
                    break
            except Exception as e:
                print(f"Error decoding chunk {chunk_idx}: {str(e)}")
        
        print(f"Decoded {len(text_chunks)} chunks containing approximately {tokens_processed} tokens")
        
    except Exception as e:
        print(f"Error using np.memmap: {str(e)}")
    
    return text_chunks

def create_embeddings(model, texts, device, output_file, chunk_size=CHUNK_SIZE):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with h5py.File(output_file, 'w') as f:
        embedding_dim = model.get_sentence_embedding_dimension()
        
        embeddings_dataset = f.create_dataset(
            'embeddings', 
            shape=(0, embedding_dim), 
            maxshape=(None, embedding_dim),
            chunks=(chunk_size, embedding_dim),
            dtype='f'
        )
        
        current_position = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                embeddings = model.encode(batch, convert_to_numpy=True)
                
                batch_size = embeddings.shape[0]
                embeddings_dataset.resize(current_position + batch_size, axis=0)
                
                embeddings_dataset[current_position:current_position + batch_size] = embeddings
                current_position += batch_size

def main():
    parser = argparse.ArgumentParser(description='Create embeddings for ClimbLab training data.')
    parser.add_argument('--clusters', nargs='+', default=CLUSTERS, 
                        help='Clusters to process (space-separated list)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_PATH,
                        help='Directory to save embeddings')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS,
                        help='Maximum number of tokens to process per cluster')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='Name of the sentence-transformer model to use')
    parser.add_argument('--skip_to_cluster', type=str, default=None,
                        help='Skip to this cluster (useful for resuming after errors)')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Size of each text chunk in tokens')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.model_name} model...")
    
    model = SentenceTransformer(args.model_name)
    
    device = torch.device(args.device)
    model.to(device)
    print(f"Model loaded. Using device: {device}")
    
    skip_mode = args.skip_to_cluster is not None
    
    for cluster in args.clusters:
        if skip_mode and cluster != args.skip_to_cluster:
            print(f"Skipping {cluster} as per --skip_to_cluster argument")
            continue
        elif skip_mode and cluster == args.skip_to_cluster:
            print(f"Resuming from {cluster}")
            skip_mode = False
        
        output_file = os.path.join(args.output_dir, f"{cluster}_train_embeddings.h5")
        
        if os.path.exists(output_file):
            print(f"Skipping {cluster} as {output_file} already exists")
            continue
            
        try:
            print(f"\nProcessing cluster: {cluster}")
            
            text_chunks = load_and_decode_climblab_data(cluster, args.max_tokens, args.chunk_size)
            
            if not text_chunks:
                print(f"No valid text chunks found for {cluster}, skipping")
                continue
            
            print(f"Generating embeddings and saving to {output_file}")
            create_embeddings(model, text_chunks, device, output_file, chunk_size=CHUNK_SIZE)
            
            print(f"Completed processing for {cluster}")
            
            del text_chunks
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing cluster {cluster}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nEmbedding generation complete!")

if __name__ == "__main__":
    main()