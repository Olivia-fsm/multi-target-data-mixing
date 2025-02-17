import os
import torch
import torch.nn as nn
import torch.optim as optim

# MLP for regression
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
        
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 64], output_dim=1, dropout_rate=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights)
    
    def forward(self, x):
        return self.network(x)
    
    