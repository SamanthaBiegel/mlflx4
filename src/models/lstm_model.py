# This module defines two LSTM-based neural networks for GPP prediction (regression). 
# The first model takes input numerical sequences (x) to predict an output sequence (y). 
# The ModelCond network architecture is conditioned on categorical features (c)
# in addition.

import torch.nn as nn
import torch
import torch.nn.init as init


class ModelCond(nn.Module):
    def __init__(self, input_dim, conditional_dim, hidden_dim, num_layers=2):
        super().__init__()

        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.3, batch_first=True)

        # Categorical features are used (vegetation class and land use), concatenated to the hidden state
        self.fc1 = nn.Sequential(
        nn.Linear(in_features=hidden_dim+conditional_dim, out_features=64),
        nn.ReLU()
        )
        # Fully connected layers for feature processing
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU()
        )
        self.fc3= nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        
        # Final linear layer for regression output
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x, c):
        # Forward pass through the LSTM layer
        out, (h,d) = self.lstm(x)

        # Concatenate conditional features
        out = torch.cat([out,c], dim=2)
        
        # Pass the concatenated output through fully connected layers
        y = self.fc1(out)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)

        return y



class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        current_dim = hidden_dim

        # layers = []
        
        # # Create additional layers that half the dimension each time until it reaches 16
        # while current_dim > 16:
        #     next_dim = max(current_dim // 2, 16)
        #     layer = nn.Sequential(
        #         nn.Linear(current_dim, next_dim),
        #         # nn.BatchNorm1d(next_dim),
        #         nn.ReLU()
        #     )
        #     layers.append(layer)
        #     current_dim = next_dim
        
        # # Save all layers in an nn.ModuleList
        # self.layers = nn.ModuleList(layers)
        
        # Final linear layer for regression output to 1
        self.final_layer = nn.Linear(current_dim, 1)
        
    def forward(self, x):

        # Forward pass through the LSTM layer
        x, (h,d) = self.lstm(x)

        # Pass the output through the rest of the layers
        # for layer in self.layers:
        #     batch_size, seq_length, feature_dim = x.shape
        #     x = x.reshape(-1, feature_dim)
        #     x = layer(x)
        #     feature_dim = x.shape[-1]
        #     x = x.reshape(batch_size, seq_length, feature_dim)

        batch_size, seq_length, feature_dim = x.shape
        x = x.reshape(-1, feature_dim)
        x = self.final_layer(x)
        x = x.reshape(batch_size, seq_length, -1)

        return x
