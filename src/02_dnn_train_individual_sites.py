# This is the final dnn model with leave-one-site-out cross-validation

# Custom modules and functions
from models.dnn_model import Model
from data.dataloader import compute_center, gpp_dataset_onesite
from utils.utils import set_seed, generate_filename
from utils.train_test_loops import test_loop
from utils.train_model import train_model
from utils.evaluate_model import evaluate_model
from sklearn.metrics import r2_score, root_mean_squared_error
import torch.optim as optim

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-device', '--device', default='cuda:0' ,type=str, help='Indices of GPU to enable')
parser.add_argument('-e', '--n_epochs', default=150, type=int, help='Number of training epochs')
parser.add_argument('-es', '--early_stopping', default=True, type=bool, help='Whether to use early stopping')
parser.add_argument('-p', '--patience', default=10, type=int, help='Number of iterations (patience threshold) used for early stopping')
parser.add_argument('-hd', '--hidden_size', default=32, type=int, help='Size of the hidden layer of the DNN model')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training the model')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate for the optimizer')
args = parser.parse_args()

print("Starting one-site overfitting on DNN model:")
print(f"> Device: {args.device}")
print(f"> Epochs: {args.n_epochs}")
if args.early_stopping:
    print(f"> Early stopping after {args.patience} epochs without improvement")
print(f"> Hidden size (DNN): {args.hidden_size}")
print(f"> Batch size: {args.batch_size}")
print(f"> Learning rate: {args.learning_rate}")

set_seed(40)

# Read data, including variables for stratified train-test split
data = pd.read_csv('../data/processed/fdk_v3_ml.csv', index_col='sitename', parse_dates=['TIMESTAMP'])

site_list = ["FR-Pue", "US-How", "IT-Cpz", "IT-SRo", "US-PFa", "US-WCr", "FI-Hyy"]
data = data[data.index.isin(site_list)]

# Get data dimensions to match DNN model dimensions
INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai']).shape[1]

# Constants
base_path = "../models"

# Generate filename
hparams = {"n_epochs": args.n_epochs, "early_stopping": args.early_stopping, "patience": args.patience, "hidden_units": args.hidden_size, "batch_size": args.batch_size, "learning_rate": args.learning_rate}
filename = generate_filename("dnn", "singlesites", hparams)

all_metrics = {'r2': [], 'rmse': [], 'nmae': [], 'abs_bias': []}

dfs_out = []

for i, (sitename, site_data) in enumerate(data.groupby(data.index)):
    site_data['year'] = site_data['TIMESTAMP'].dt.year
    years = site_data['year'].unique()
    
    for year in years[1:]:
        test_years = [year]
        train_years = years[years < year]
        data_test = site_data[site_data['year'].isin(test_years)]
        data_train = site_data[site_data['year'].isin(train_years)]
        data_train = data_train.drop(columns = ['year'])
        data_test = data_test.drop(columns = ['year'])

        # Calculate mean and standard deviation to normalize the data
        train_mean, train_std = compute_center(data_train)

        # Format pytorch dataset for the data loader
        # Normalize training and validation data according to the training center
        train_ds = gpp_dataset_onesite(data_train, train_mean, train_std)
        val_ds = gpp_dataset_onesite(data_train, train_mean, train_std)

        train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
        val_dl = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False)

        ## Define the model to be trained
        # Initialise the DNN model, set layer dimensions to match data
        model = Model(input_dim = INPUT_FEATURES, hidden_dim=args.hidden_size).to(device = args.device)

        # Initialise the optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9)

        # Initiate tensorboard logging instance for this site
        writer = SummaryWriter(log_dir = f"{base_path}/runs/{filename}/site_{sitename}")
        if i == 0:
            print(f"Logging to {base_path}/runs/{filename}")

        ## Train the model

        # Return best validation R2 score and the center used to normalize training data (repurposed for testing on leaf-out site)
        best_val_metrics, model = train_model(train_dl, val_dl,
                                                    model, optimizer, scheduler, writer,
                                                    args.n_epochs, args.device, args.patience, args.early_stopping)
        
        print(f"Validation scores for site {sitename}: R2 = {best_val_metrics['r2']:.4f} | RMSE = {best_val_metrics['rmse']:.4f} | NMAE = {best_val_metrics['nmae']:.4f} | Abs Bias = {best_val_metrics['abs_bias']:.4f}")

        # Stop logging, for this site
        writer.close()

        ## Model evaluation

        # Format pytorch dataset for the data loader
        test_ds = gpp_dataset_onesite(data_test, train_mean, train_std, test = False)
        test_dl = DataLoader(test_ds, batch_size = 1, shuffle = False)

        # Evaluate model on test set
        test_loss, y_pred = test_loop(test_dl, model, args.device)

        test_metrics, data_test_eval = evaluate_model(test_dl, y_pred)

        print(f"Test scores for site {sitename}, year {year}: R2 = {test_metrics['r2']:.4f} | RMSE = {test_metrics['rmse']:.4f} | NMAE = {test_metrics['nmae']:.4f} | Abs Bias = {test_metrics['abs_bias']:.4f}")

        for key in test_metrics.keys():
            all_metrics[key].append(test_metrics[key])

        dfs_out.append(data_test_eval[['TIMESTAMP', 'GPP_NT_VUT_REF', 'gpp_pred']])

        print("")

df_out = pd.concat(dfs_out)

print(f"DNN - Mean R2: {np.mean(all_metrics['r2']):.4f} | Mean RMSE: {np.mean(all_metrics['rmse']):.4f} | Mean NMAE: {np.mean(all_metrics['nmae']):.4f} | Mean Abs Bias: {np.mean(all_metrics['abs_bias']):.4f}")

preds_filename = f"{base_path}/preds/{filename}.csv"
df_out.to_csv(preds_filename)

print(f"Predictions saved to {preds_filename}")