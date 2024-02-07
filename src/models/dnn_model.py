# This module defines a PyTorch model for GPP predictions (regression task). 
# It consists of a number of fully connected layers with ReLU activation functions.
# The architecture matches the fully connected layers, with same dimensions,
# that are defined on top of the LSTM cells in lstm_model.py

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        current_dim = hidden_dim

        # First layer maps from input_dim to hidden_dim
        layers = [nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )]
        
        # Create additional layers that half the dimension each time until it reaches 16
        while current_dim > 16:
            next_dim = max(current_dim // 2, 16)
            layer = nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.ReLU()
            )
            layers.append(layer)
            current_dim = next_dim
        
        # Save all layers in an nn.ModuleList
        self.layers = nn.ModuleList(layers)
        
        # Final linear layer for regression output to 1
        self.final_layer = nn.Linear(current_dim, 1)
        
    def forward(self, x):
        # Define the forward pass through the neural network
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x