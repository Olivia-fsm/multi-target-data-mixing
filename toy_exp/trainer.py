import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, TensorDataset
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from model import MLP
from data_utils import WeightedDataLoader, IndividualDataLoader, ToyDataArguments

@dataclass
class ToyTrainArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lr_end: float = field(
            default=1e-4,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    lr_scheduler_name: str = field(
        default='linear_warmup_cosine', metadata={"help": "Custom LR scheduler name (linear_warmup_exponential, linear_warmup_cosine, linear_cooldown)"}
    )
    reweight_train: str = field(
        default="mix", metadata={"help": "Reweighting training domains. [mix: change sampling; grad: manipulate graduents; None: fixed.]"}
    )
    reweight_tgt: str = field(
        default="mix", metadata={"help": "Reweighting target domains. [mix: change sampling; grad: manipulate graduents; None: fixed.]"}
    )

class ToyTrainer:
    def __init__(self, 
                 model: nn.Module,
                 train_loader: IndividualDataLoader,
                 tgt_loader: IndividualDataLoader,
                 val_domains: List[str],
                 val_loader_ls: List[DataLoader],
                 mix_train_loader: WeightedDataLoader,
                 mix_tgt_loader: WeightedDataLoader,
                 loss_fn: nn.Module = nn.MSELoss(),
                 lr: float = 1e-3,
                 scheduler_type: str = 'cosine',
                 scheduler_params: dict = {},
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 log_steps: int = 1000,
                 eval_steps: int = 1000,
                 save_steps: int = 5000,
                 output_dir: str = None,
                 ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.tgt_loader = tgt_loader
        self.val_loader_ls = val_loader_ls
        self.val_domains = val_domains  # val task names
        self.mix_train_loader = mix_train_loader
        self.mix_tgt_loader = mix_tgt_loader
        self.loss_fn = loss_fn
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.device = device
        self.output_dir = output_dir        
        os.makedirs(output_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': {task:[] for task in val_domains},
            'avg_val_loss': [],
            'learning_rates': []
        }        
        self.best_val_loss = float('inf')
        
    def get_optimizer_and_scheduler(self, num_steps):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        
        if self.scheduler_type == 'cosine':
            # Standard cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_steps,
                eta_min=self.scheduler_params.get('eta_min', 1e-6)
            )
        elif self.scheduler_type == 'linear':
            # Linear scheduler
            scheduler = LinearLR(
                optimizer,
                start_factor=self.scheduler_params.get('start_factor', 1.0),
                end_factor=self.scheduler_params.get('end_factor', 0.1),
                total_iters=num_steps,
            )
        elif self.scheduler_type == 'cosine_warmup':
            # Cosine annealing with warm restarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_params.get('T_0', num_steps),  # Initial restart interval
                T_mult=self.scheduler_params.get('T_mult', 2),  # Multiply factor for restart interval
                eta_min=self.scheduler_params.get('eta_min', 1e-6)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
            
        return optimizer, scheduler
    
    def get_batch(self, data_loader):
        """Get next batch, reinitialize iterator if needed"""
        for batch in data_loader:
            return batch
    
    def train_step(self, X, y):
        """Perform single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        X, y = X.to(self.device), y.to(self.device)        
        outputs = self.model(X)
        loss = self.loss_fn(outputs, y)        
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, num_steps):
        log_interval = self.log_steps
        checkpoint_interval = self.save_steps
        eval_interval = self.eval_steps
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(num_steps)
        """Train for specified number of steps"""
        for step in tqdm(range(num_steps)):
            # Get batch and train
            batch = self.get_batch(self.mix_train_loader)
            X, y = batch
            loss = self.train_step(X, y)
            
            # Update learning rate and record history
            self.scheduler.step()
            self.history['train_loss'].append(loss)
            self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            # Evaluate and save ckpt
            if (step + 1) % eval_interval == 0:
                avg_loss, val_losses = self.eval_all()

                is_best = avg_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_loss

                if (step + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(step + 1, avg_loss, is_best)
            
            # Log progress
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(self.history['train_loss'][-log_interval:])
                print(f"Step [{step+1}/{num_steps}] train-loss: {avg_loss:.4f} "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        # Plot training progress
        self.plot_training_progress()
        return self.history
    
    @torch.no_grad()
    def eval(self, val_iter):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for X, y in val_iter:
            X = X.to(self.device)
            y = y.to(self.device)
            
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def eval_all(self):
        val_iters = [iter(dl) for dl in self.val_loader_ls]
        val_losses = []
        for val_iter, val_name in zip(val_iters, self.val_domains):
            val_loss = self.eval(val_iter)
            self.history["val_loss"][val_name].append(val_loss)
            val_losses.append(val_loss)
        avg_loss = np.mean(val_losses)
        self.history["avg_val_loss"].append(avg_loss)
        return avg_loss, val_losses
    
    def update_train_weights(self):
        pass
    
    def update_tgt_weights(self):
        pass
    
    def save_checkpoint(self, step, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Save latest checkpoint
        filename = f'checkpoint_step_{step}.pt'
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['step']
    
    def plot_training_progress(self):
        """Plot training progress including loss curves and learning rate schedule"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
        
        # Plot losses
        steps = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(steps, self.history['train_loss'], label='Train Loss')
        ax1_2 = ax1.twinx()
        
        # Plot validation loss at evaluation points
        eval_steps = range(len(self.history['avg_val_loss']))
        eval_indices = [i * len(steps) // len(eval_steps) for i in eval_steps]
        ax1_2.plot(eval_indices, self.history['avg_val_loss'], label='Valid[average]', marker='o', color="orange")
        
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss', color='b')
        ax1_2.set_ylabel('Average Valid Loss', color='orange')
        ax1.set_title('Training and Validation Loss')
        # ax1.legend()
        ax1.grid(True)
        
        ax2_2 = ax2.twinx()
        val_axes = [ax2, ax2_2]
        colors = ["blue", "orange"]
        idx = 0
        for val_domain, val_loss in self.history['val_loss'].items():
            val_axes[idx].plot(eval_indices, val_loss, label=f'Valid[{val_domain}]', marker='o', color=colors[idx])
            idx += 1
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Val Loss[L1]', color='b')
        ax2_2.set_ylabel('Val Loss[L2]', color='orange')
        ax2.set_title('Validation Loss')
        # ax2.legend()
        # ax2_2.legend()
        ax2.grid(True)
        
        # Plot learning rate schedule
        ax3.plot(steps, self.history['learning_rates'])
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'training_progress.png'))
        plt.close()
        

if __name__ == '__main__':
    from data_utils import interleave_dataloader, individual_dataloader, load_data
    
    train_weights = [0., 0., 0., 1.0, 0., 0.]
    tgt_weights = [0.5, 0.5]
    valid_weights = [0.2,0.2]
    data_config = ToyDataArguments()
    train_dataset_ls, tgt_dataset_ls, val_dataset_ls = load_data(data_config)
    mix_train_loader = interleave_dataloader(train_dataset_ls, train_weights,
                          batch_size = 64,
                          num_worker = 0)
    mix_tgt_loader = interleave_dataloader(tgt_dataset_ls, tgt_weights,
                          batch_size = 64,
                          num_worker = 0)
    train_loader = individual_dataloader(train_dataset_ls, batch_size = 4, num_worker = 0)
    tgt_loader = individual_dataloader(tgt_dataset_ls, batch_size = 4, num_worker = 0)
    valid_loaders = [DataLoader(val_ds, batch_size = 4, num_workers = 0) for val_ds in val_dataset_ls]
    
    for batch in mix_train_loader:
        print(batch)
        break
    
    for batch in train_loader:
        print(batch)
        break
    
    print("=========== Trainer ===========")
    model = MLP(hidden_dims=[128, 64, 64, 64, 32],
                dropout_rate=0.1)
    train_args = ToyTrainArguments()
    trainer = ToyTrainer(model,
                         lr = 1e-4,
                 train_loader = train_loader,
                 tgt_loader = tgt_loader,
                 val_domains = ["L1", "L2"],
                 val_loader_ls = valid_loaders,
                 mix_train_loader = mix_train_loader,
                 mix_tgt_loader = mix_tgt_loader,
                 log_steps = 1000,
                 eval_steps = 1000,
                 save_steps = 50000,
                 output_dir="/scratch/homes/sfan/multi_doge/toy_exp/exp/dummy")
    trainer.train(num_steps=50000)
    # import pdb
    # pdb.set_trace()