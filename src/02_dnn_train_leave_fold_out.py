# This is the final DNN model with leave-one-site-out cross-validation

# Custom modules and functions
from models.dnn_model import Model
from data.dataloader import gpp_dataset, compute_center
from utils.utils import set_seed
from utils.train_test_loops import train_loop, test_loop
from utils.train_model import train_model
from utils.train_test_split import train_test_split_chunks
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, root_mean_squared_error

# Load necessary dependencies
import argparse
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch


# Parse arguments 
parser = argparse.ArgumentParser(description='CV DNN')

parser.add_argument('-device', '--device', default='cuda:0' ,type=str,
                      help='Indices of GPU to enable')

parser.add_argument('-e', '--n_epochs', default=150, type=int,
                      help='Number of training epochs (per site, for the leave-site-out CV)')

parser.add_argument('-o', '--output_file', default='', type=str,
                    help='File name to save output')

parser.add_argument('-p', '--patience', default=10, type=int,
                    help='Number of iterations (patience threshold) used for early stopping')

parser.add_argument('-b', '--batch_size', default=16, type=int,
                    help='Batch size for training the model')

parser.add_argument('-d', '--hidden_dim', default=256, type=int,
                    help='Hidden dimension of the DNN model')

parser.add_argument('-l', '--learning_rate', default=0.01, type=float,
                    help='Learning rate for the optimizer')

args = parser.parse_args()

# Set random seeds for reproducibility
set_seed(40)

print("Starting leave-site-out on DNN model:")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")
print(f"> Early stopping after {args.patience} epochs without improvement")

# Read imputed data, including variables for stratified train-test split and imputation flag
data = pd.read_csv('../data/processed/df_imputed.csv', index_col=0)

# Create list of sites for leave-site-out cross validation
sites = data.index.unique()

# Get data dimensions to match DNN model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai', 'chunk_id']).shape[1]

def generate_filename(base_path, n_epochs, patience, batch_size, learning_rate, hidden_units):
        # Format learning rate to avoid using '.' in file names
        lr_formatted = f"{learning_rate:.0e}".replace('-', 'm').replace('.', 'p')
        
        # Get current date and time
        current_datetime = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        
        # Generate filename
        filename = (f"dnn_lfo_alldata_epochs{n_epochs}_patience{patience}"
                        f"_bs{batch_size}_lr{lr_formatted}_hu{hidden_units}"
                        f"_{current_datetime}")
        return f"{base_path}/{filename}"

# Constants
base_path = "../models"

# Generate filename
filename = generate_filename(base_path, args.n_epochs, args.patience, args.batch_size, args.learning_rate, args.hidden_dim)

# Process and save predictions dataframe
df_out = data.loc[:, ['TIMESTAMP','GPP_NT_VUT_REF']].copy()

# Initialise data.frame to store GPP predictions, from the trained DNN model
y_pred_sites = {}

# Group by 'sitename' and calculate mean temperature and aridity
grouped = data.groupby('sitename').agg({'TA_F_MDS': 'mean', 'ai': 'first'})

grouped = grouped.dropna(subset=["ai"])

# Discretize numerical columns into bins
grouped['TA_F_MDS_bins'] = pd.qcut(grouped['TA_F_MDS'], 2, labels=False).astype(str)
grouped['ai_bins'] = pd.qcut(grouped['ai'], 2, labels=False).astype(str)

# Combine discretized columns into a single categorical column for stratification
grouped['combined_target'] = grouped['TA_F_MDS_bins'] + '_' + grouped['ai_bins']

all_r2 = []
all_rmse = []

kf = StratifiedKFold(n_splits=5, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(grouped.index, grouped['combined_target'])):
    train_sites = grouped.index.unique()[train_index]
    test_sites = grouped.index.unique()[test_index]
    
    data_train_val = data[data.index.isin(train_sites)]
    data_test = data[data.index.isin(test_sites)]

    # Separate train-val split
    data_train, data_val, chunks_train, chunks_val = train_test_split_chunks(data_train_val)

    # Calculate mean and standard deviation to normalize the data
    train_mean, train_std = compute_center(data_train)

    # Format pytorch dataset for the data loader
    # Normalize training and validation data according to the training center
    train_ds = gpp_dataset(data_train, train_mean, train_std)
    val_ds = gpp_dataset(data_val, train_mean, train_std)

    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False)

    ## Define the model to be trained
    # Initialise the DNN model, set layer dimensions to match data
    model = Model(input_dim = INPUT_FEATURES, hidden_dim=args.hidden_dim).to(device = args.device)

    # Initialise the optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initiate tensorboard logging instance for this site
    if len(args.output_file) == 0:
        writer = SummaryWriter(log_dir = f"../models/runs/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}/fold_{i+1}")
    else:
        writer = SummaryWriter(log_dir = f"../models/runs/{args.output_file}/fold_{i+1}")

    ## Train the model

    # Return best validation R2 score and the center used to normalize training data (repurposed for testing on leaf-out site)
    best_r2, best_rmse, model = train_model(train_dl, val_dl,
                                                model, optimizer, writer,
                                                args.n_epochs, args.device, args.patience)
    
    print(f"Validation scores for fold {i+1}: R2 = {best_r2:.4f} | RMSE = {best_rmse:.4f}")

    # Save model weights from best epoch
    # if len(args.output_file)==0:
    #     torch.save(model,
    #         f = f"../models/weights/dnn_lso_epochs_{args.n_epochs}_patience_{args.patience}_fold_{i+1}.pt")
    # else:
    #     torch.save(model, f = f"../models/weights/{args.output_file}_fold_{i+1}.pt")

    # Stop logging, for this site
    writer.close()

    ## Model evaluation

    # Format pytorch dataset for the data loader
    test_ds = gpp_dataset(data_test, train_mean, train_std, test = False)

    # Run data loader with batch_size = 1
    # Due to different time series lengths per site,
    # we cannot load several sites per batch
    test_dl = DataLoader(test_ds, batch_size = 1, shuffle = False)

    # Evaluate model on test set, removing imputed GPP values
    test_loss, test_r2, test_rmse, y_pred = test_loop(test_dl, model, args.device)

    data_test_eval = data_test.copy()
    data_test_eval['gpp_pred'] = [item for sublist in y_pred for item in sublist]

    # Filter inputs for nans
    nan_y_true = data_test_eval["GPP_NT_VUT_REF"].isna()
    nan_y_pred = data_test_eval["gpp_pred"].isna()
    data_test_eval = data_test_eval[~(nan_y_true | nan_y_pred)]

    r2_test = r2_score(y_true = data_test_eval["GPP_NT_VUT_REF"], y_pred = data_test_eval["gpp_pred"])
    rmse_test = root_mean_squared_error(y_true = data_test_eval["GPP_NT_VUT_REF"], y_pred = data_test_eval["gpp_pred"])

    for j, s in enumerate(data_test.index.unique()):
        y_pred_sites[s] = y_pred[j]

    print(f"Test scores for fold {i+1}: R2 = {r2_test:.4f} | RMSE = {rmse_test:.4f}")

    all_r2.append(r2_test)
    all_rmse.append(rmse_test)

    print("")

print(f"DNN - Mean R2: {np.mean(all_r2):.4f} | Mean RMSE: {np.mean(all_rmse):.4f}")

for s in y_pred_sites.keys():
    df_out.loc[[i == s for i in df_out.index], 'gpp_dnn'] = np.asarray(y_pred_sites.get(s))

preds_filename = f"{filename}.csv".replace(base_path, f"{base_path}/preds")
df_out.to_csv(preds_filename)

print(f"Predictions saved to {preds_filename}")