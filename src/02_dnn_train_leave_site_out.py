# This is the final DNN model with leave-one-site-out cross-validation

# Custom modules and functions
from models.dnn_model import Model
from data.preprocess import compute_center
from data.dataloader import gpp_dataset
from utils.utils import set_seed
from utils.train_test_loops import train_loop, test_loop

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

from plotly import graph_objects as go



# Parse arguments 
parser = argparse.ArgumentParser(description='CV DNN')

parser.add_argument('-device', '--device', default='cuda:0' ,type=str,
                      help='Indices of GPU to enable')

parser.add_argument('-e', '--n_epochs', default=150, type=int,
                      help='Number of training epochs (per site, for the leave-site-out CV)')

parser.add_argument('-o', '--output_file', default='', type=str,
                    help='File name to save output')

args = parser.parse_args()


# Set random seeds for reproducibility
set_seed(40)

print("Starting leave-site-out on DNN model:")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")

# Read imputed and raw data
data = pd.read_csv('../data/processed/df_imputed.csv', index_col=0)
raw = pd.read_csv('../data/raw/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']

# Remove site with too little observations
raw = raw[raw.index != 'CN-Cng']

# Create list of sites for leave-site-out cross validation
sites = raw.index.unique()

# Get data dimensions to match LSTM model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = 'GPP_NT_VUT_REF').shape[1]

# Initialise data.frame to store GPP predictions, from the trained LSTM model
y_pred_sites = {}

# Loop over all sites, 
# An LSTM model is trained on all sites except the "left-out-site"
# for a given number of epochs
for s in sites:
    print(f"Test Site: {s}")

    # Split data (numerical time series) for leave-site-out cross validation
    # A single site is kept for testing and all others are used for training
    data_train = data.loc[ data.index != s ]
    data_test = data.loc[ data.index == s]

    # Get a mask to discard imputed testing values in the model evaluation call
    mask_imputed = [ not m for m in raw.loc[ raw.index == s].isna() ]

    # Calculate mean and standard deviation to center the data
    train_mean, train_std = compute_center(data_train)

    # Format pytorch dataset for the data loader
    train_ds = gpp_dataset(data_train, train_mean, train_std)
    test_ds = gpp_dataset(data_test, train_mean, train_std)

    # Run data loader with batch_size = 1
    # Due to different time series lengths per site,
    # we cannot load several sites per batch
    train_dl = DataLoader(train_ds, batch_size = 1, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = 1, shuffle = True)

    ## Define model to be trained

    # Initialise the LSTM model, set layer dimensions to match data
    model = Model(input_dim = INPUT_FEATURES).to(device = args.device)

    # Initialise the optimiser
    optimizer = torch.optim.Adam(model.parameters())

    # Start recording R2 score after each epoch, initialise at -Inf
    r2 = -np.Inf

    # Initiate tensorboard logging instance for this site
    writer = SummaryWriter(log_dir = "../model/runs", comment = s)

    # Train the model
    for epoch in range(args.n_epochs):
        
        # Perform one round of training, doing backpropagation for each training site
        # Obtain the cumulative MSE (training loss) and R2
        train_loss, train_r2 = train_loop(train_dl, model, optimizer,
                                          args.device)

        

        # Log tensorboard training values
        writer.add_scalar("mse_loss/train", train_loss, epoch)
        writer.add_scalar("r2_mean/train", train_r2, epoch)         # summed R2, will not be in [0,1] 

        # Evaluate model on test set, removing imputed GPP values
        test_loss, test_r2, y_pred = test_loop(test_dl, model, mask_imputed,
                                               args.device)
        
        # Log tensorboard testing values
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("R2/test", test_r2, epoch)

        # Save the prediction for the best epoch, based on the test R2
        if test_r2 >= r2:
            y_pred_sites[s] = y_pred

            # Update best R2 score
            r2 = test_r2

        # TODO: Early stop

    print(f"R2 score for site {s}:")
    print(r2)

    # Save model weights from last epoch
    if len(args.output_file)==0:
        torch.save(model,
            f = f"../model/weights/dnn_lso_epochs_{args.n_epochs}_{s}.pt")
    else:
        torch.save(model, f = f"../model/weights/{args.output_file}_{s}.pt")

    # Stop logging, for this site
    writer.close()

# Save predictions into a data.frame. aligning with raw data
df_out = pd.read_csv('../data/raw/df_20210510.csv', index_col=0)[['date', 'GPP_NT_VUT_REF']]
df_out = df_out[df_out.index != 'CN-Cng']

for s in df_out.index.unique():
    df_out.loc[[i == s for i in df_out.index], 'gpp_lstm'] = np.asarray(y_pred_sites.get(s))

# Save to a csv, to be processed in R
if len(args.output_file)==0:
    df_out.to_csv(f"../model/preds/dnn_lso_epochs_{args.n_epochs}.csv")   
else:
    df_out.to_csv("../model/preds/" + args.output_file)