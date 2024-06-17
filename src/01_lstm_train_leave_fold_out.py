# LSTM model with leave-fold-out cross-validation

# Custom modules and functions
from models.lstm_model import Model
from data.dataloader import gpp_dataset, compute_center
from utils.utils import set_seed
from utils.train_test_loops import train_loop, test_loop
from utils.train_model import train_model
from utils.train_test_split import train_test_split_sites
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, root_mean_squared_error
import torch.optim as optim

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-device', '--device', default='cuda:0' ,type=str, help='Indices of GPU to enable')
parser.add_argument('-e', '--n_epochs', default=500, type=int, help='Number of training epochs')
parser.add_argument('-es', '--early_stopping', default=True, type=bool, help='Whether to use early stopping')
parser.add_argument('-p', '--patience', default=50, type=int, help='Number of iterations (patience threshold) used for early stopping')
parser.add_argument('-hd', '--hidden_size', default=512, type=int, help='Size of the hidden layer of the LSTM model')
parser.add_argument('-l', '--num_layers', default=4, type=int, help='Number of layers for the LSTM model')
parser.add_argument('-d', '--dropout', default=0.4, type=float, help='Dropout rate for the LSTM model')
parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size for training the model')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate for the optimizer')
parser.add_argument('-sp', '--scheduler_patience', default=30, type=int, help='Patience for the learning rate scheduler')
parser.add_argument('-sf', '--scheduler_factor', default=0.1, type=float, help='Factor for the learning rate scheduler')
args = parser.parse_args()

print("Starting leave-fold-out training and validation on LSTM model:")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")
if args.early_stopping:
    print(f"> Early stopping after {args.patience} epochs without improvement")
print(f"> Hidden size (LSTM): {args.hidden_size}")
print(f"> Number of layers (LSTM): {args.num_layers}")
if args.num_layers > 1:
    print(f"> Dropout rate (LSTM): {args.dropout}")
else:
     args.dropout = 0
print(f"> Batch size: {args.batch_size}")
print(f"> Learning rate: {args.learning_rate}")
print(f"> Learning rate scheduler: ReduceLROnPlateau(patience={args.scheduler_patience}, factor={args.scheduler_factor})")

set_seed(40)

# Read data, including variables for stratified train-test split
data = pd.read_csv('../data/processed/fdk_v3_ml.csv', index_col='sitename', parse_dates=['TIMESTAMP'])

# Create list of sites for leave-site-out cross validation
sites = data.index.unique()

# Get data dimensions to match LSTM model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai']).shape[1]

def generate_filename(n_epochs, patience, batch_size, learning_rate, hidden_units):
        # Format learning rate to avoid using '.' in file names
        lr_formatted = f"{learning_rate:.0e}".replace('-', 'm').replace('.', 'p')
        dropout_formatted = str(args.dropout).replace('.', 'p')
        
        # Get current date and time
        current_datetime = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        
        # Generate filename
        filename = (f"lstm_lfo_alldata_epochs{n_epochs}_patience{patience}"
                        f"_bs{batch_size}_lr{lr_formatted}_hu{hidden_units}"
                        f"_dropout{dropout_formatted}_nlayers{args.num_layers}"
                        f"_{current_datetime}")
        return f"{filename}"

# Constants
base_path = "../models"

# Generate filename
filename = generate_filename(args.n_epochs, args.patience, args.batch_size, args.learning_rate, args.hidden_size)

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

dfs_out = []

kf = StratifiedKFold(n_splits=5, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(grouped.index, grouped['combined_target'])):
    train_sites = grouped.index.unique()[train_index]
    test_sites = grouped.index.unique()[test_index]
    
    data_train_val = data[data.index.isin(train_sites)]
    data_test = data[data.index.isin(test_sites)]

    # Separate train-val split
    data_train, data_val, sites_train, sites_val = train_test_split_sites(data_train_val)

    # Calculate mean and standard deviation to normalize the data
    train_mean, train_std = compute_center(data_train)

    # Format pytorch dataset for the data loader
    # Normalize training and validation data according to the training center
    train_ds = gpp_dataset(data_train, train_mean, train_std)
    val_ds = gpp_dataset(data_val, train_mean, train_std)

    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False)

    ## Define the model to be trained
    # Initialise the LSTM model, set layer dimensions to match data
    model = Model(input_dim = INPUT_FEATURES, hidden_dim=args.hidden_size, dropout=args.dropout, num_layers=args.num_layers).to(device = args.device)

    # Initialise the optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.scheduler_patience, factor=args.scheduler_factor)

    # Initiate tensorboard logging instance for this fold
    writer = SummaryWriter(log_dir = f"{base_path}/runs/{filename}/fold_{i+1}")
    if i == 0:
        print(f"Logging to {base_path}/runs/{filename}")

    ## Train the model

    # Return best validation R2 score and the center used to normalize training data (repurposed for testing on leaf-out site)
    best_r2, best_rmse, model = train_model(train_dl, val_dl,
                                                 model, optimizer, scheduler, writer,
                                                 args.n_epochs, args.device, args.patience, args.early_stopping)
    
    print(f"Validation scores for fold {i+1}: R2 = {best_r2:.4f} | RMSE = {best_rmse:.4f}")
    
    # Save model weights from best epoch
    torch.save(model, f = f"{base_path}/weights/{filename}_fold_{i+1}.pt")

    # Stop logging, for this site
    writer.close()

    ## Model evaluation

    # Format pytorch dataset for the data loader
    test_ds = gpp_dataset(data_test, train_mean, train_std, test = False)
    test_dl = DataLoader(test_ds, batch_size = 1, shuffle = False)

    # Evaluate model on test set
    test_loss, y_pred = test_loop(test_dl, model, args.device)

    data_test_eval = data_test.copy()
    data_test_eval['gpp_pred'] = [item for sublist in y_pred for item in sublist]

    # filter inputs for nans
    nan_y_true = data_test_eval["GPP_NT_VUT_REF"].isna()
    nan_y_pred = data_test_eval["gpp_pred"].isna()
    data_test_eval = data_test_eval[~(nan_y_true | nan_y_pred)]

    r2_test = r2_score(y_true = data_test_eval["GPP_NT_VUT_REF"], y_pred = data_test_eval["gpp_pred"])
    rmse_test = root_mean_squared_error(y_true = data_test_eval["GPP_NT_VUT_REF"], y_pred = data_test_eval["gpp_pred"])

    dfs_out.append(data_test_eval[['TIMESTAMP', 'GPP_NT_VUT_REF', 'gpp_pred']])

    print(f"Test scores for fold {i+1}: R2 = {r2_test:.4f} | RMSE = {rmse_test:.4f}")

    all_r2.append(r2_test)
    all_rmse.append(rmse_test)

    print("")

print(f"LSTM - Mean R2: {np.mean(all_r2):.4f} | Mean RMSE: {np.mean(all_rmse):.4f}") 

df_out = pd.concat(dfs_out)

preds_filename = f"{base_path}/preds/{filename}.csv"
df_out.to_csv(preds_filename)

print(f"Predictions saved to {preds_filename}")