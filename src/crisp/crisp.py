import h5py
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import argparse
import umap


def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        return f['embeddings'][:]

# def extract_subset_name(file_path):
#     filename = os.path.basename(file_path)
#     subset_name = filename.replace('_train_embeddings.h5', '')
#     generalist_subsets = ["arxiv", "book", "c4", "cc", "github", "stackexchange", "wikipedia"]
#     return subset_name if subset_name in generalist_subsets else None


# Clustering functions
def compute_subset_clusters(embeddings_dir, generalist_subsets, k1=32, use_minibatch=True, save_dir="./crisp_results"):
    subset_models = {}
    subset_centers = {}
    
    for subset in tqdm(generalist_subsets, desc="Clustering subsets"):
        embedding_file = os.path.join(embeddings_dir, f"{subset}_train_embeddings.h5")
        print(embedding_file)
        if not os.path.exists(embedding_file):
            print(f"Warning: Could not find embeddings for subset {subset}")
            continue
            
        embeddings = load_embeddings(embedding_file)
        
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=k1, random_state=42, batch_size=1024, n_init=3)
        else:
            kmeans = KMeans(n_clusters=k1, random_state=42, n_init=10)
        
        print(f"Clustering {subset} with {len(embeddings)} embeddings into {k1} clusters")
        kmeans.fit(embeddings)
        
        subset_models[subset] = kmeans
        subset_centers[subset] = kmeans.cluster_centers_
    
    return subset_models, subset_centers


def compute_nearest_clusters_distribution(embeddings, subset_centers, k2=5):
    all_centers = []
    center_subset_map = []
    
    for subset, centers in subset_centers.items():
        all_centers.append(centers)
        center_subset_map.extend([subset] * len(centers))
    
    all_centers = np.vstack(all_centers)
    center_subset_map = np.array(center_subset_map)
    
    nn = NearestNeighbors(n_neighbors=k2)
    nn.fit(all_centers)
    
    distances, indices = nn.kneighbors(embeddings)
    nearest_subsets = center_subset_map[indices]
    
    subset_counts = Counter()
    
    for embedding_idx in range(len(embeddings)):
        for neighbor_idx in range(k2):
            subset = nearest_subsets[embedding_idx, neighbor_idx]
            weight = 1.0 / (distances[embedding_idx, neighbor_idx] + 1e-5)
            subset_counts[subset] += weight
    
    total_weight = sum(subset_counts.values())
    subset_distribution = {subset: count/total_weight for subset, count in subset_counts.items()}
    
    return subset_distribution


# Visualization functions
def plot_subset_distributions(specialist_distributions, subset_names, figsize=(20, 15), save_path=None):
    n_domains = len(specialist_distributions)
    n_cols = min(3, n_domains)
    n_rows = (n_domains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_domains > 1 else [axes]
    
    for i, (domain, distribution) in enumerate(specialist_distributions.items()):
        full_distribution = {subset: distribution.get(subset, 0) for subset in subset_names}
        sorted_items = sorted(full_distribution.items(), key=lambda x: x[1], reverse=True)
        sorted_subsets = [item[0] for item in sorted_items]
        sorted_weights = [item[1] for item in sorted_items]
        
        y_pos = np.arange(len(sorted_subsets))
        axes[i].barh(y_pos, sorted_weights, align='center')
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(sorted_subsets)
        axes[i].invert_yaxis()
        axes[i].set_title(f'{domain}')
        axes[i].set_xlabel('Probability')
        
        for j, v in enumerate(sorted_weights):
            axes[i].text(v + 0.01, j, f"{v:.3f}", va='center')
    
    for i in range(n_domains, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('SlimPajama Subset Distributions for Specialist Domains', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_dimensionality_reducer(method):
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42), "PCA"
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42, perplexity=30), "t-SNE"
    elif method.lower() == 'umap':
        return umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1), "UMAP"
    else:
        raise ValueError("Method must be 'umap', 'pca', or 'tsne'")


def visualize_combined_clusters(subset_centers, specialist_embeddings, domain_name,
                              method='umap', sample_size=5000, 
                              figsize=(15, 12), save_path=None,
                              include_generalist_embeddings=False, 
                              embeddings_dir=None):
    all_centers = []
    center_labels = []
    
    for i, (subset, centers) in enumerate(subset_centers.items()):
        all_centers.append(centers)
        center_labels.extend([f"{subset}_{j}" for j in range(len(centers))])
    
    all_centers = np.vstack(all_centers)
    
    if len(specialist_embeddings) > sample_size:
        indices = np.random.choice(len(specialist_embeddings), sample_size, replace=False)
        specialist_sample = specialist_embeddings[indices]
    else:
        specialist_sample = specialist_embeddings
    
    if include_generalist_embeddings and embeddings_dir:
        generalist_samples = {}
        subset_sample_size = sample_size // len(subset_centers)
        
        for subset in subset_centers.keys():
            embedding_file = os.path.join(embeddings_dir, f"{subset}_train_embeddings.h5")
            if os.path.exists(embedding_file):
                try:
                    with h5py.File(embedding_file, 'r') as f:
                        total_embeddings = f['embeddings'].shape[0]
                        if total_embeddings > subset_sample_size:
                            indices = np.sort(np.random.choice(total_embeddings, subset_sample_size, replace=False))
                            embeddings = f['embeddings'][indices]
                        else:
                            embeddings = f['embeddings'][:]
                        generalist_samples[subset] = embeddings
                except Exception as e:
                    print(f"Error loading generalist embeddings for {subset}: {e}")
        
        combined_data = [all_centers, specialist_sample]
        for subset, samples in generalist_samples.items():
            combined_data.append(samples)
        combined_data = np.vstack(combined_data)
        
        centers_end = len(all_centers)
        specialist_end = centers_end + len(specialist_sample)
        subset_ends = {}
        current_end = specialist_end
        for subset, samples in generalist_samples.items():
            current_end += len(samples)
            subset_ends[subset] = (specialist_end, current_end)
            specialist_end = current_end
    else:
        combined_data = np.vstack([all_centers, specialist_sample])
    
    reducer, title_method = get_dimensionality_reducer(method)
    reduced_data = reducer.fit_transform(combined_data)
    
    centers_reduced = reduced_data[:len(all_centers)]
    specialist_reduced = reduced_data[len(all_centers):len(all_centers)+len(specialist_sample)]
    
    plt.figure(figsize=figsize)
    
    unique_subsets = list(set([label.split('_')[0] for label in center_labels]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subsets)))
    color_map = {subset: colors[i] for i, subset in enumerate(unique_subsets)}
    
    if include_generalist_embeddings and embeddings_dir and len(generalist_samples) > 0:
        for i, subset in enumerate(generalist_samples.keys()):
            if subset in subset_ends:
                start, end = subset_ends[subset]
                subset_reduced = reduced_data[start:end]
                plt.scatter(subset_reduced[:, 0], subset_reduced[:, 1], 
                          c=[color_map[subset]], alpha=0.2, s=3, 
                          label=f'{subset} embeddings')
    
    plt.scatter(specialist_reduced[:, 0], specialist_reduced[:, 1], 
               c='gray', alpha=0.5, s=5, label=f'{domain_name} embeddings')
    
    for i, (label, point) in enumerate(zip(center_labels, centers_reduced)):
        subset = label.split('_')[0]
        if subset in color_map:
            first_occurrence = next((j for j, lbl in enumerate(center_labels) 
                                  if lbl.split('_')[0] == subset), None)
            
            if i == first_occurrence:
                plt.scatter(point[0], point[1], c=[color_map[subset]], s=100, marker='*', 
                           label=f'{subset} centers')
            else:
                plt.scatter(point[0], point[1], c=[color_map[subset]], s=100, marker='*')
    
    plt.title(f'Cluster Centers vs {domain_name} Embeddings ({title_method})')
    plt.xlabel(f'{title_method} Component 1')
    plt.ylabel(f'{title_method} Component 2')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if len(by_label) > 10:
        filtered_labels = {}
        for k, v in by_label.items():
            subset = k.split(' ')[0] if ' ' in k else None
            if subset in unique_subsets or k == f'{domain_name} embeddings':
                filtered_labels[k] = v
        plt.legend(filtered_labels.values(), filtered_labels.keys(), loc='best', fontsize='small')
    else:
        plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_all_specialists_combined(subset_centers, specialist_domains, embeddings_dir,
                                      method='umap', sample_size=2000, 
                                      figsize=(20, 16), save_path=None,
                                      include_generalist_embeddings=False):
    n_domains = len(specialist_domains)
    n_cols = min(3, n_domains)
    n_rows = (n_domains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    reducer, title_method = get_dimensionality_reducer(method)
    
    all_centers = []
    center_labels = []
    
    for subset, centers in subset_centers.items():
        all_centers.append(centers)
        center_labels.extend([f"{subset}_{j}" for j in range(len(centers))])
    
    all_centers = np.vstack(all_centers)
    
    unique_subsets = list(set([label.split('_')[0] for label in center_labels]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subsets)))
    color_map = {subset: colors[i] for i, subset in enumerate(unique_subsets)}
    
    generalist_samples = {}
    if include_generalist_embeddings:
        subset_sample_size = sample_size // len(subset_centers) 
        for subset in subset_centers.keys():
            embedding_file = os.path.join(embeddings_dir, f"{subset}_train_embeddings.h5")
            if os.path.exists(embedding_file):
                try:
                    with h5py.File(embedding_file, 'r') as f:
                        total_embeddings = f['embeddings'].shape[0]
                        if total_embeddings > subset_sample_size:
                            indices = np.sort(np.random.choice(total_embeddings, subset_sample_size, replace=False))
                            embeddings = f['embeddings'][indices]
                        else:
                            embeddings = f['embeddings'][:]
                        generalist_samples[subset] = embeddings
                except Exception as e:
                    print(f"Error loading generalist embeddings for {subset}: {e}")
    
    for i, domain in enumerate(tqdm(specialist_domains, desc="Creating plots")):
        if i >= len(axes):
            print(f"Warning: Not enough subplots for domain {domain}")
            continue
            
        embedding_file = os.path.join(embeddings_dir, f"{domain}_train_embeddings.h5")
        
        if not os.path.exists(embedding_file):
            print(f"Warning: Could not find embeddings for domain {domain}")
            axes[i].text(0.5, 0.5, f"No data for {domain}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(domain)
            continue
            
        embeddings = load_embeddings(embedding_file)
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            specialist_sample = embeddings[indices]
        else:
            specialist_sample = embeddings
        
        if include_generalist_embeddings and generalist_samples:
            combined_data = [all_centers, specialist_sample]
            for subset, samples in generalist_samples.items():
                combined_data.append(samples)
            combined_data = np.vstack(combined_data)
            
            centers_end = len(all_centers)
            specialist_end = centers_end + len(specialist_sample)
            subset_ends = {}
            current_end = specialist_end
            for subset, samples in generalist_samples.items():
                current_end += len(samples)
                subset_ends[subset] = (specialist_end, current_end)
                specialist_end = current_end
        else:
            combined_data = np.vstack([all_centers, specialist_sample])
        
        reduced_data = reducer.fit_transform(combined_data)
        centers_reduced = reduced_data[:len(all_centers)]
        specialist_reduced = reduced_data[len(all_centers):len(all_centers)+len(specialist_sample)]
        
        ax = axes[i]
        
        if include_generalist_embeddings and generalist_samples:
            for subset in generalist_samples.keys():
                if subset in subset_ends:
                    start, end = subset_ends[subset]
                    subset_reduced = reduced_data[start:end]
                    ax.scatter(subset_reduced[:, 0], subset_reduced[:, 1], 
                              c=[color_map[subset]], alpha=0.1, s=1)
        
        ax.scatter(specialist_reduced[:, 0], specialist_reduced[:, 1], 
                  c='gray', alpha=0.5, s=3, label=f'{domain}')
        
        for j, (label, point) in enumerate(zip(center_labels, centers_reduced)):
            subset = label.split('_')[0]
            if subset in color_map:
                ax.scatter(point[0], point[1], c=[color_map[subset]], s=30, marker='*')
        
        ax.set_title(domain)
        ax.set_xlabel(f'Component 1')
        ax.set_ylabel(f'Component 2')
    
    for i in range(len(specialist_domains), len(axes)):
        axes[i].axis('off')
    
    legend_elements = []
    for subset, color in color_map.items():
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=f'{subset} centers'))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='gray', markersize=5, 
                                     label='Specialist embeddings'))
    
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=len(legend_elements)//2 + 1,
              frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle(f'SlimPajama Subset Centers vs Specialist Domains ({title_method})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, axes


def main(embeddings_dir, generalist_subsets, specialist_domains, k1=32, k2=5, 
         save_dir="./crisp_results", include_generalist_in_plots=False,
         plot_method='umap'):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Computing clusters for each subset with k1={k1}")
    _, subset_centers = compute_subset_clusters(
        embeddings_dir, generalist_subsets, k1, True, save_dir
    )
    
    print(f"Computing specialist distributions with k2={k2}")
    specialist_distributions = {}
    
    for domain in tqdm(specialist_domains, desc="Processing specialist domains"):
        embedding_file = os.path.join(embeddings_dir, f"{domain}_train_embeddings.h5")
        
        if not os.path.exists(embedding_file):
            print(f"Warning: Could not find embeddings for domain {domain}")
            continue
            
        embeddings = load_embeddings(embedding_file)
        distribution = compute_nearest_clusters_distribution(embeddings, subset_centers, k2)
        specialist_distributions[domain] = distribution
    
    print(f"Creating combined visualization using {plot_method}")
    combined_viz_path = os.path.join(save_dir, f"all_domains_combined_{plot_method}_k1_{k1}.png")
    visualize_all_specialists_combined(
        subset_centers, specialist_domains, embeddings_dir,
        method=plot_method, figsize=(20, 16), save_path=combined_viz_path,
        include_generalist_embeddings=include_generalist_in_plots
    )
    
    distributions_file = os.path.join(save_dir, f"specialist_subset_distributions_k1_{k1}_k2_{k2}.json")
    with open(distributions_file, 'w') as f:
        json.dump(specialist_distributions, f, indent=2)
    
    csv_file = os.path.join(save_dir, f"specialist_subset_distributions_k1_{k1}_k2_{k2}.csv")
    with open(csv_file, 'w') as f:
        f.write("domain," + ",".join(generalist_subsets) + "\n")
        for domain, dist in specialist_distributions.items():
            row = [domain]
            for subset in generalist_subsets:
                row.append(str(dist.get(subset, 0.0)))
            f.write(",".join(row) + "\n")
    
    plot_path = os.path.join(save_dir, f"subset_distributions_k1_{k1}_k2_{k2}.png")
    plot_subset_distributions(
        specialist_distributions, generalist_subsets, 
        figsize=(20, 15), save_path=plot_path
    )
    
    print(f"Analysis complete! Saved distributions to {distributions_file} and plots to {save_dir}")
    return specialist_distributions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze embeddings and create visualizations')
    # wiki40b or slimpajama
    parser.add_argument('--dataset', type=str, default='slimpajama',
                        help='Dataset to analyze (slimpajama or wiki40b)')
    parser.add_argument('--embeddings_dir', type=str, default="data/dataset_embeddings",
                        help='Directory containing embedding files')
    parser.add_argument('--k1', type=int, default=5, 
                        help='Number of clusters per subset')
    parser.add_argument('--k2', type=int, default=5, 
                        help='Number of nearest neighbors to consider')
    parser.add_argument('--save_dir', type=str, default="./crisp_results",
                        help='Directory to save results')
    parser.add_argument('--include_generalist', action='store_true', 
                        help='Include generalist embeddings in plots')
    parser.add_argument('--plot_method', type=str, default="pca", choices=['umap', 'pca', 'tsne'],
                        help='Method for dimensionality reduction in plots')
    
    args = parser.parse_args()
    dataset_embeddings_dir = os.path.join(args.embeddings_dir, args.dataset)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset)
    
    
    if args.dataset == 'wiki40b':
        generalist_subsets = ["en", "fr", "de", "es", "it", "ru"] 
        specialist_domains = ["pl", "uk", "nl", "pt", "ca", "tr", "da","ro"]
    elif args.dataset == 'slimpajama':
        generalist_subsets = ["arxiv", "book", "c4", "cc", "github", "stackexchange", "wikipedia"]
        specialist_domains = ["arc_challenge", "gsm8k", "arc_easy", "hellaswag", "piqa", "sciq", "kodcode", "logiqa"]
    main(
        dataset_embeddings_dir, generalist_subsets, specialist_domains,
        k1=args.k1, k2=args.k2, save_dir=dataset_save_dir,
        include_generalist_in_plots=args.include_generalist,
        plot_method=args.plot_method
    )
