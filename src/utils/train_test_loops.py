# This file contains the training and testing
# functions used in the main scripts

# Import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd

def custom_loss(outputs, targets, masks):
    # Use the mask to zero out contributions from padded values
    loss = (outputs - targets) ** 2  # MSE loss
    masked_loss = loss * masks  # Apply the mask
    return masked_loss.sum() / masks.sum()  # Normalize by the number of non-padded values

def train_loop(dataloader, model, optimizer, DEVICE, writer, steps, cat=False):

    # Initiate training loss, to aggregate over sites
    train_loss = 0.0

    # Set model to training mode (activates dropout, batch normalization, etc, if present)
    model.train()

    n_batches = len(dataloader)

    loss = nn.MSELoss()

    # Loop over all training sites
    for x, y in dataloader:
        # Send tensors to the correct device
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        # Perform forward pass for predictions
        y_pred = model(x)

        # Reset gradient to zero, rather than aggregating gradient over sites
        optimizer.zero_grad()

        # Compute MSE loss between GPP and predicted GPP
        output = loss(y_pred.flatten(), y.flatten())

        # Backpropagate the gradients through the model
        output.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Store gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_scalar(f"gradients/{name}", param.grad.norm(), steps)

        # Update model parameters using the optimizer
        optimizer.step()

        # Accumulate the training loss over sites
        train_loss += output.item()

    # Set model to evaluation mode (deactivate dropout, etc)
    model.eval()

    # Return computed training loss
    return train_loss/n_batches, model



def train_loop_cat(dataloader, model, optimizer, DEVICE):

    # Initiate training loss, to aggregate over sites
    train_loss = 0.0

    # Set model to training mode (activates dropout, batch normalization, etc, if present)
    model.train()

    n_batches = len(dataloader)

    # Loop over all training sites
    for x, y, c, padding_mask in dataloader:

        # Send tensors to the correct device
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        c = c.to(DEVICE)
        
        # Perform forward pass for predictions
        y_pred = model(x, c)

        # Reset gradient to zero, rather than aggregating gradient over sites
        optimizer.zero_grad()

        # Compute MSE losss between GPP and predicted GPP
        loss = custom_loss(y_pred.flatten(), y.flatten(), padding_mask.flatten())

        # Backpropagate the gradients through the model
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        # Accumulate the training loss over sites
        train_loss += loss.item()

    # Set model to evaluation mode (deactivate dropout, etc)
    model.eval()

    # Return computed training loss
    return train_loss/n_batches


def test_loop(dataloader, model, DEVICE):

    # Set model to evaluation mode
    model.eval()

    # Initiate testing loss, to aggregate over sites (if there are more than one)
    test_loss = 0.0

    n_batches = len(dataloader)

    all_y_pred = []

    loss = nn.MSELoss()
    count_preds = 0

    # Stop computing gradients during the following code chunk
    with torch.no_grad():
        # Get testing data from dataloader
        for x, y in dataloader:

            # Send tensors to the correct device
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Perform forward pass for predictions
            y_pred = model(x)

            # Compute test MSE on testing data
            test_loss += loss(y_pred.squeeze().flatten(), y.squeeze().flatten())
            count_preds += y_pred.squeeze().flatten().shape[0]

            all_y_pred.append(y_pred.squeeze().detach().cpu().numpy())       

    # Return computed testing loss
    return test_loss/n_batches, all_y_pred


def test_loop_cat(dataloader, model, DEVICE):

    # Set model to evaluation mode
    model.eval()

    # Initiate testing losses, to aggregate over sites (if there are more than one)
    test_loss = 0.0
    
    n_batches = len(dataloader)

    # Stop computing gradients during the following code chunk:d
    with torch.no_grad():

        # Get testing data from dataloader
        for x, y, c, padding_mask in dataloader:

            # Send tensors to the correct device
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            c = c.to(DEVICE)

            # Perform forward pass for predictions
            y_pred = model(x, c)

            # Squeeze empty tensor dimensions to match y and y_pred
            y = y.squeeze()
            y_pred = y_pred.squeeze()

            # Compute test MSE on testing data
            test_loss += custom_loss(y_pred.flatten(), y.flatten(), padding_mask.flatten())

            # Transform prediction tensor into numpy array
            y_pred = y_pred.detach().cpu().numpy()
    
    # Return computed testing loss
    return test_loss/n_batches, y_pred
