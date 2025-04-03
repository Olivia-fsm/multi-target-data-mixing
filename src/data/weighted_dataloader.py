import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import List, Callable, Optional, Iterator
from itertools import chain

class WeightedDataset(Dataset):
    """
    A dataset that combines multiple datasets with their respective lengths
    """
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumsum_lengths = np.cumsum([0] + self.lengths)
    
    def __len__(self) -> int:
        return sum(self.lengths)
    
    def __getitem__(self, idx: int):
        # Find which dataset the index belongs to
        dataset_idx = np.searchsorted(self.cumsum_lengths[1:], idx, side='right')
        # Get the local index within that dataset
        local_idx = idx - self.cumsum_lengths[dataset_idx]
        # import pdb
        # pdb.set_trace()
        return self._get_dataset_item(self.datasets[dataset_idx], local_idx)
    
    def _get_dataset_item(self, local_dataset, local_idx: int):
        return {k: local_dataset[k][local_idx] for k in local_dataset.features}

class WeightedRandomSampler(Sampler[int]):
    """
    Samples elements from [0,...,len(weights)-1] with given probabilities (weights)
    Supports sampling more than 2^24 samples by chunking
    """
    def __init__(self, weights: List[float], lengths: List[int], num_samples: Optional[int] = None):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float64)
        
        if len(weights) != len(lengths):
            raise ValueError("Number of weights must match number of datasets")
            
        self.weights = weights / weights.sum()  # Normalize weights
        self.lengths = lengths
        self.num_samples = num_samples if num_samples is not None else sum(lengths)
        
        # Create cumulative lengths for mapping global indices to local indices
        self.cumsum_lengths = np.cumsum([0] + lengths)
        
        # Maximum number of samples for torch.multinomial (2^24)
        self.MAX_SAMPLES = 2**24 - 1
        
    def __iter__(self) -> Iterator[int]:
        remaining_samples = self.num_samples
        
        while remaining_samples > 0:
            # Calculate number of samples for this chunk
            chunk_size = min(remaining_samples, self.MAX_SAMPLES)
            
            # Sample dataset indices according to weights for this chunk
            dataset_indices = torch.multinomial(
                self.weights,
                chunk_size,
                replacement=True
            )
            
            # For each selected dataset in this chunk, sample a random index
            for dataset_idx in dataset_indices:
                start_idx = self.cumsum_lengths[dataset_idx]
                dataset_length = self.lengths[dataset_idx]
                # Generate random index within the selected dataset
                local_idx = torch.randint(dataset_length, (1,)).item()
                # Convert to global index
                yield start_idx + local_idx
            
            remaining_samples -= chunk_size
    
    def __len__(self) -> int:
        return self.num_samples

    def update_weights(self, new_weights: List[float]) -> None:
        """
        Update sampling weights
        Args:
            new_weights: New sampling weights for each dataset
        """
        if len(new_weights) != len(self.lengths):
            raise ValueError("Number of new weights must match number of datasets")
        
        if not isinstance(new_weights, torch.Tensor):
            new_weights = torch.tensor(new_weights, dtype=torch.float64)
            
        self.weights = new_weights / new_weights.sum()  # Normalize new weights

class WeightedDataLoader(DataLoader):
    """
    DataLoader that performs weighted sampling across multiple datasets
    """
    def __init__(
        self,
        dataset_ls: List[Dataset],
        weights: List[float],
        batch_size: int,
        data_collator: Optional[Callable] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        # Combine datasets
        dataset = WeightedDataset(dataset_ls)
        
        # Create sampler
        self.sampler = WeightedRandomSampler(
            weights=weights,
            lengths=[len(ds) for ds in dataset_ls],
            num_samples=num_samples
        )
        
        # Initialize parent DataLoader
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=data_collator,
            **kwargs
        )
        
        # Store number of datasets for validation
        self.num_datasets = len(dataset_ls)
    
    def update_weights(self, new_weights: List[float]) -> None:
        """
        Update the sampling weights for each dataset
        Args:
            new_weights: List of new sampling weights for each dataset
        """
        if len(new_weights) != self.num_datasets:
            raise ValueError(f"Expected {self.num_datasets} weights, got {len(new_weights)}")
        self.sampler.update_weights(new_weights)