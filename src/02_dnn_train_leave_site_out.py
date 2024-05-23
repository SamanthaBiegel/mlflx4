# This is the final DNN model with leave-one-site-out cross-validation

# Custom modules and functions
from models.dnn_model import Model
from data.dataloader import gpp_dataset, compute_center
from utils.utils import set_seed
from utils.train_test_loops import train_loop, test_loop
from utils.train_model import train_model
from utils.train_test_split import train_test_split_chunks

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


# Parse arguments 
parser = argparse.ArgumentParser(description='CV DNN')

parser.add_argument('-device', '--device', default='cuda:0' ,type=str,
                      help='Indices of GPU to enable')

parser.add_argument('-e', '--n_epochs', default=500, type=int,
                      help='Number of training epochs (per site, for the leave-site-out CV)')

parser.add_argument('-o', '--output_file', default='', type=str,
                    help='File name to save output')

parser.add_argument('-p', '--patience', default=50, type=int,
                    help='Number of iterations (patience threshold) used for early stopping')

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size for training the model')

parser.add_argument('-d', '--hidden_dim', default=512, type=int,
                    help='Hidden dimension of the DNN model')

parser.add_argument('-l', '--learning_rate', default=0.005, type=float,
                    help='Learning rate for the optimizer')

args = parser.parse_args()

# Set random seeds for reproducibility
set_seed(40)

print("Starting leave-site-out on DNN model:")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")
print(f"> Early stopping after {args.patience} epochs without improvement")

# Read data, including variables for stratified train-test split
data = pd.read_csv('../data/processed/fdk_v3_ml.csv', index_col=0)

# Create list of sites for leave-site-out cross validation
sites = data.index.unique()

# Get data dimensions to match LSTM model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai', 'chunk_id']).shape[1]

# Initialise data.frame to store GPP predictions, from the trained LSTM model
y_pred_sites = {}

# Loop over all sites, 
# An LSTM model is trained on all sites except the "left-out-site"
# for a given number of epochs
for s in sites:

    # Split data (numerical time series) for leave-site-out cross validation
    # A single site is kept for testing and all others are used for training
    data_train_val = data.loc[ data.index != s ]
    data_test = data.loc[ data.index == s]

    data_train_val = data_train_val.dropna(subset=["chunk_id"])

    # Separate train-val split
    data_train, data_val, chunks_train, chunks_val = train_test_split_chunks(data_train_val)

    # Calculate mean and standard deviation to normalize the data
    train_mean, train_std = compute_center(data_train)

    # Format pytorch dataset for the data loader
    # Normalize training and validation data according to the training center
    train_ds = gpp_dataset(data_train, train_mean, train_std)
    val_ds = gpp_dataset(data_val, train_mean, train_std)

    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = args.batch_size, shuffle = True)

    ## Define the model to be trained
    # Initialise the DNN model, set layer dimensions to match data
    model = Model(input_dim = INPUT_FEATURES, hidden_dim=args.hidden_dim).to(device = args.device)

    # Initialise the optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initiate tensorboard logging instance for this site
    if len(args.output_file) == 0:
        writer = SummaryWriter(log_dir = f"../models/runs/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}/{s}")
    else:
        writer = SummaryWriter(log_dir = f"../models/runs/{args.output_file}/{s}")

    ## Train the model

    # Return best validation R2 score and the center used to normalize training data (repurposed for testing on leaf-out site)
    best_r2, best_mae, model = train_model(train_dl, val_dl,
                                                 model, optimizer, writer,
                                                 args.n_epochs, args.device, args.patience)
    
    print(f"Validation scores for site {s}: R2 = {best_r2:.4f} | MAE = {best_mae:.4f}")

    # Save model weights from best epoch
    if len(args.output_file)==0:
        torch.save(model,
            f = f"../models/weights/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}_{s}.pt")
    else:
        torch.save(model, f = f"../models/weights/{args.output_file}_{s}.pt")

    # Stop logging, for this site
    writer.close()


    ## Model evaluation

    # Format pytorch dataset for the data loader
    test_ds = gpp_dataset(data_test, train_mean, train_std, test = False)

    # Run data loader with batch_size = 1
    # Due to different time series lengths per site,
    # we cannot load several sites per batch
    test_dl = DataLoader(test_ds, batch_size = 1, shuffle = True)

    # Evaluate model on test set
    test_loss, test_r2, test_mae, y_pred = test_loop(test_dl, model, args.device)

    # Save prediction for the left-out site
    y_pred_sites[s] = y_pred

    print(f"Test scores for site {s}: R2 = {test_r2:.4f} | MAE = {test_mae:.4f}")
    print("")

    

# Save predictions into a data.frame. aligning with raw data
df_out = data[['TIMESTAMP', 'GPP_NT_VUT_REF']].copy()

for s in df_out.index.unique():
    y_pred = y_pred_sites.get(s)
    if y_pred is None:
        y_pred = np.nan  # np.nan is compatible with float32
    else:
        # Ensure the array is of dtype float32
        y_pred = np.asarray(y_pred, dtype=np.float32)

    df_out.loc[[i == s for i in df_out.index], 'gpp_dnn'] = y_pred

# Save to a csv, to be processed in R
if len(args.output_file)==0:
    df_out.to_csv(f"../models/preds/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}_batch_{args.batch_size}_hidden_{args.hidden_dim}_lr_{args.learning_rate}.csv")  
else:
    df_out.to_csv("../models/preds/" + args.output_file)