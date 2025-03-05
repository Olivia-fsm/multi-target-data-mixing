import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset

# Define dataset save path
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATASET_DIR, exist_ok=True)

# Define your functions
def f1(x):
    return np.maximum(np.tanh(0.5 * x[1]), 0)

def f2(x):
    return np.maximum(np.tanh(-0.5 * x[1]), 0)

def g1(x):
    return np.log(np.maximum(np.abs(0.5 * (-x[0] - 7) - np.tanh(-x[1])), 0.000005)) + 6

def g2(x):
    return np.log(np.maximum(np.abs(0.5 * (-x[0] + 3) - np.tanh(-x[1]) + 2), 0.000005)) + 6

def g3(x):
    return np.log(np.maximum(np.abs(0.5 * (-x[0] - 7) - np.tanh(x[1] + 2) + 2), 0.000005)) + 6

def g4(x):
    return np.log(np.maximum(np.abs(0.5 * (-x[0] + 3) - np.tanh(x[1] + 2) + 2), 0.000005)) + 6

def h1(x):
    return ((-x[0] + 7)**2 + 0.1 * (-x[0] - 8)**2) / 10 - 20

def h2(x):
    return ((-x[0] - 7)**2 + 0.1 * (-x[0] - 8)**2) / 10 - 20

def h3(x):
    return ((-x[0] + 7)**2 + 0.1 * (-x[0] + 8)**2) / 10 - 20

def h4(x):
    return ((-x[0] - 7)**2 + 0.1 * (-x[0] + 8)**2) / 10 - 20

def L1(x):
    return 0.1 * (f1(x) * g1(x) + f2(x) * h1(x))

def L2(x):
    return f1(x) * g2(x) + f2(x) * h2(x)

def L3(x):
    return 0.5 * (f1(x) * g3(x) + f2(x) * h3(x))

def L4(x):
    return 0.25 * (f1(x) * g4(x) + f2(x) * h4(x))

def noise_func(y_min: float, y_max: float, size: int = 1):
    subset = [np.random.sample() * (y_max-y_min) + y_min for _ in range(size)]
    return subset 

def get_train_val_indices(num_sample_train=100000, num_sample_tgt=1000, x_min=-10, x_max=10):
    # generate train indices
    x1 = np.random.uniform(x_min, x_max, num_sample_train)
    x2 = np.random.uniform(x_min, x_max, num_sample_train)
    X_train = np.column_stack((x1, x2))
    
    # generate target indices
    x1 = np.random.uniform(x_min, x_max, num_sample_tgt)
    x2 = np.random.uniform(x_min, x_max, num_sample_tgt)
    X_tgt = np.column_stack((x1, x2))
    
    # generate valid indices
    x1 = np.linspace(x_min, x_max, 100)
    np.random.shuffle(x1)
    x2 = np.linspace(x_min, x_max, 100)
    np.random.shuffle(x2)
    X1, X2 = np.meshgrid(x1, x2)
    X_val = np.column_stack((X1.flatten(), X2.flatten()))
    return X_train, X_tgt, X_val

def get_train(tgt_func, X_train, num_subset_train=1000,):
    np.random.shuffle(X_train)
    X_subset = np.array(X_train)[np.random.choice(range(len(X_train)), size=num_subset_train, replace=False)]
    Y_subset = np.array([tgt_func(x) for x in X_subset])
    return X_subset, Y_subset
    
def get_val(tgt_func, X_val, data_path=f"{DATASET_DIR}/dummy/val.pt"):
    data_dir = "/".join(data_path.split("/")[:-1])
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(data_path):
        print(f"🐤 Load Validation Set from {data_path}!")
        return torch.load(data_path, weights_only=False)
    Y_val = np.array([tgt_func(x) for x in X_val])
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    dataset = TensorDataset(X_val, Y_val)
    torch.save(dataset, data_path)
    ds_type = "Validation" if "val" in data_path else "Train"
    print(f"🐣 New Target {ds_type} Set saved to {data_path}!")
    return dataset

def get_train_domain(tgt_func_ls: list,
                     tgt_ratios: list,
                     tgt_X_train: list,
                     subset_size: int = 1000,
                     data_path: str = f"{DATASET_DIR}/dummy/train.pt"):
    data_dir = "/".join(data_path.split("/")[:-1])
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(data_path):
        print(f"🐤 Load Domain Dataset from {data_path}!")
        return torch.load(data_path, weights_only=False)
    assert len(tgt_func_ls)==len(tgt_ratios), "Lengths of ratios and target functions are not matched!"
    assert sum(tgt_ratios)<=1.0, "Sum of all ratios cannot exceed 1.0!"
    func_sizes = [int(r*subset_size) for r in tgt_ratios]
    
    X, Y = [], []
    for f,x,s in zip(tgt_func_ls, tgt_X_train, func_sizes):
        func_X, func_Y = get_train(f, x, s)
        X.extend(func_X)
        Y.extend(func_Y)
            
    if sum(tgt_ratios) < 1.0:
        noise_ratio = 1.0 - sum(tgt_ratios)
        noise_size = int(noise_ratio*subset_size)
        func_X = np.array(tgt_X_train[-1])[np.random.choice(range(len(tgt_X_train[-1])), noise_size, replace=False)]
        func_Y = noise_func(y_min=np.min(Y), y_max=np.max(Y), size=noise_size)
        X.extend(func_X)
        Y.extend(func_Y)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X, Y)
    torch.save(dataset, data_path)
    print(f"🐣 New Domain Dataset saved to {data_path}!")
    config_path = data_path.replace("train", "config")
    torch.save(torch.tensor(tgt_ratios, dtype=torch.float32), config_path)
    return dataset

def get_datasets(ratios: list,
                 tgt_funcs: list = [L1, L2, L3],
                 x_min = -10, x_max = 10,
                 num_sample_train = 100000,
                 num_sample_tgt = 5000,
                 subset_size = 5000):
    rst_dict = {
        "train": {},
        "val": {},
    }
    
    tgt_X_train = []
    # generate validation datasets
    for idx,f in enumerate(tgt_funcs, start=1):
        # generate indices splits
        X_train, X_tgt, X_val = get_train_val_indices(num_sample_train=num_sample_train, num_sample_tgt=num_sample_tgt, x_min=x_min, x_max=x_max)
        train = get_val(f, X_tgt, data_path=f"{DATASET_DIR}/L{idx}/train.pt")
        val = get_val(f, X_val, data_path=f"{DATASET_DIR}/L{idx}/val.pt")
        rst_dict["train"][f"L{idx}"] = train
        rst_dict["val"][f"L{idx}"] = val        
        tgt_X_train.append(X_train)
    
    for idx, ratio in enumerate(ratios, start=1):
        train = get_train_domain(tgt_func_ls=tgt_funcs,
                                tgt_ratios=ratio,
                                tgt_X_train=tgt_X_train,
                                subset_size=subset_size,
                                data_path=f"{DATASET_DIR}/D{idx}/train.pt")
        rst_dict["train"][f"D{idx}"] = train
    return rst_dict

    
def main():
     # Generate the datasets or load them if they exist
    ratios = [(0.75, 0.15, 0.1, 0.), (0.1, 0.75, 0.15, 0.), (0.15, 0.1, 0.75, 0.), 
              (1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.)]
    dataset = get_datasets(ratios=ratios,
                 tgt_funcs = [L1, L2, L3, L4],
                 x_min = -10, x_max = 10,
                 num_sample_train = 1000000,
                 num_sample_tgt = 5000,
                 subset_size = 100000)

    train_dataset_D1 = dataset['train']['D1']
    train_dataset_D2 = dataset['train']['D2']
    train_dataset_D3 = dataset['train']['D3']
    train_dataset_D4 = dataset['train']['D4']
    train_dataset_D5 = dataset['train']['D5']
    train_dataset_D6 = dataset['train']['D6']
    val_dataset_L1 = dataset['val']['L1']
    val_dataset_L2 = dataset['val']['L2']
    val_dataset_L3 = dataset['val']['L3']
    val_dataset_L4 = dataset['val']['L4']

    # Print first 5 training samples
    print("Train (D1):")
    for i in range(5):
        print(train_dataset_D1[i])  # Each sample is a (X, Y) tuple

    # Print first 5 validation samples for L1
    print("\nValidation (L1):")
    for i in range(5):
        print(val_dataset_L1[i])

    # Print first 5 validation samples for L2
    print("\nValidation (L2):")
    for i in range(5):
        print(val_dataset_L2[i])
        
    # Print first 5 validation samples for L2
    print("\nValidation (L3):")
    for i in range(5):
        print(val_dataset_L3[i])

    def scatter_plot(dataset, 
                    title="2D Scatter Plot",
                    cmap='viridis',
                    size_range=(15, 60),
                    alpha=0.7,
                    colorbar=True,
                    figsize=(8, 6),
                    ):
        """
        Create a scatter plot from 2D points with color and size representing a scalar value.
        
        Args:
            X: array-like of shape (n_samples, 2)
                The 2D coordinates for each point
            Y: array-like of shape (n_samples,)
                The scalar values to represent with color and size
            title: str
                Title of the plot
            cmap: str
                Colormap name from matplotlib
            size_range: tuple
                (min_size, max_size) for the scatter points
            alpha: float
                Transparency of points (0 to 1)
            colorbar: bool
                Whether to show the colorbar
            figsize: tuple
                Figure size in inches
                
        Returns:
            fig, ax: matplotlib figure and axis objects
        """
        # Convert inputs to numpy arrays
        X = [data[0].numpy() for data in dataset]  # Extract features
        Y = [data[1].numpy() for data in dataset]  # Extract targets
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize Y values for size scaling
        size_min, size_max = size_range
        if Y.max() != Y.min():
            sizes = size_min + (size_max - size_min) * (Y - Y.min()) / (Y.max() - Y.min())
        else:
            sizes = [size_min + (size_max - size_min)/2] * len(Y)
        
        # Create scatter plot
        scatter = ax.scatter(X[:, 0], X[:, 1], 
                            c=Y,
                            s=sizes,
                            cmap=cmap,
                            alpha=alpha)
        
        # Add colorbar
        if colorbar:
            plt.colorbar(scatter, label='Y Value')
        
        # Set labels and title
        ax.set_xlabel('X[0]')
        ax.set_ylabel('X[1]')
        ax.set_title(title)
        
        # Equal aspect ratio for better visualization
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"/scratch/homes/sfan/multi_doge/toy_exp/plots/{title}.png")
        return fig, ax
    
    # Plot train and validation datasets
    scatter_plot(train_dataset_D1, "Train-D1")
    scatter_plot(train_dataset_D2, "Train-D2")
    scatter_plot(train_dataset_D3, "Train-D3")
    scatter_plot(train_dataset_D4, "Train-D4")
    scatter_plot(train_dataset_D5, "Train-D5")
    scatter_plot(train_dataset_D6, "Train-D6")
    scatter_plot(val_dataset_L1, "target-L1")
    scatter_plot(val_dataset_L2, "target-L2")
    scatter_plot(val_dataset_L3, "target-L3")
    scatter_plot(val_dataset_L4, "target-L4")
    
# Execute the main function
if __name__ == '__main__':
    main()
