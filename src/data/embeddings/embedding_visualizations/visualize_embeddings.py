import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap


def load_all_embeddings(embedding_dir):
    """
    Load embeddings from all HDF5 files in the directory
    
    Args:
        embedding_dir (str): Directory containing embedding files
    
    Returns:
        tuple: (embeddings_list, dataset_names, dataset_labels)
    """
    embeddings_list = []
    dataset_names = []
    
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('_embeddings.h5')]
    
    if not embedding_files:
        raise ValueError("No embedding files found.")
    
    for embedding_file in embedding_files:
        file_path = os.path.join(embedding_dir, embedding_file)
        dataset_name = embedding_file.replace('_train_embeddings.h5', '')
        # TODO : uncomment this & change it to filter datasets
        # if dataset_name not in ["hellaswag", "arxiv","c4","cc","wikipedia","stackexchange","book","github"]:
        #     print(f"Skipping unsupported dataset: {dataset_name}")
        #     continue  
        # Load embeddings
        with h5py.File(file_path, 'r') as f:
            embeddings = f['embeddings'][:1000]
      
        embeddings_list.append(embeddings)
        dataset_names.append(dataset_name)
    
    return embeddings_list, dataset_names

def visualize_combined_embeddings(embeddings_list, dataset_names, method='umap'):
    """
    Visualize embeddings from multiple datasets in a single plot
    
    Args:
        embeddings_list (list): List of embedding arrays
        dataset_names (list): List of dataset names
        method (str): Dimensionality reduction method
    """
    combined_embeddings = np.vstack(embeddings_list)
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(combined_embeddings)
    
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        method_name = 'UMAP'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, 
                       perplexity=min(30, len(combined_embeddings) - 1))
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        method_name = 't-SNE'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        method_name = 'PCA'
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    cumulative_lengths = np.cumsum([0] + [len(emb) for emb in embeddings_list])
    
    plt.figure(figsize=(16, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(dataset_names)))
    
    for i, (dataset_name, start, end) in enumerate(zip(dataset_names, 
                                                       cumulative_lengths[:-1], 
                                                       cumulative_lengths[1:])):
        color = colors[i % len(colors)]
        plt.scatter(
            reduced_embeddings[start:end, 0], 
            reduced_embeddings[start:end, 1], 
            label=dataset_name,
            color=color,
            alpha=0.7, 
            edgecolors='black', 
            linewidth=0.5
        )
    
    plt.title(f'Combined Embeddings - {method_name} Visualization', fontsize=16)
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.legend(title='Datasets', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_dir = 'embedding_visualizations/wiki40b'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'combined_{method.lower()}_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {output_path}")

def main():
    # TODO : change this to your embedding directory
    embedding_dir = 'dataset_embeddings/wiki40b'
    
    if not os.path.exists(embedding_dir):
        print(f"Error: Directory {embedding_dir} does not exist.")
        return
    
    try:
        embeddings_list, dataset_names = load_all_embeddings(embedding_dir)
    except ValueError as e:
        print(e)
        return
    
    # NOTE : uncomment methods to visualize
    visualize_methods = ['umap']#, 'tsne', 'pca']
    
    for method in visualize_methods:
        visualize_combined_embeddings(
            embeddings_list, 
            dataset_names, 
            method=method
        )

if __name__ == '__main__':
    main()
