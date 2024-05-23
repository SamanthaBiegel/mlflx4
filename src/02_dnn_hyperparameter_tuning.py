# Custom modules and functions
from models.dnn_model import Model
from utils.utils import set_seed
from utils.train_model import train_model
from data.dataloader import gpp_dataset, compute_center
from utils.train_test_split import train_test_split_chunks
from torch.utils.data import DataLoader

# Load necessary dependencies
import argparse
import numpy as np
import pandas as pd
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Parse arguments 
parser = argparse.ArgumentParser(description='DNN Hyperparameter Tuning')

parser.add_argument('-device', '--device', default='cuda:0', type=str, help='Indices of GPU to enable')
parser.add_argument('-e', '--n_epochs', default=150, type=int, help='Number of training epochs')
parser.add_argument('-p', '--patience', default=10, type=int, help='Number of iterations (patience threshold) for early stopping')
parser.add_argument('-t', '--num_trials', default=10, type=int, help='Number of trials for hyperparameter tuning')

args = parser.parse_args()

# Set random seeds for reproducibility
set_seed(40)

# Load and preprocess the entire dataset
data = pd.read_csv('../data/processed/fdk_v3_ml.csv', index_col="sitename", parse_dates=['TIMESTAMP'])

# Hyperparameter tuning setup
batch_sizes_list = [16, 32, 64, 128, 256]
hidden_dim_list = [32, 64, 128, 256, 512]
learning_rates_list = [1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4]

best_model = None
best_hyperparameters = None
best_validation_score = np.inf

# Tensorboard writer setup (optional)
writer = SummaryWriter(log_dir=f"../models/runs/hyperparameter_tuning")

INPUT_FEATURES = data.select_dtypes(include = ['int', 'float']).drop(columns = ['GPP_NT_VUT_REF', 'ai', 'chunk_id']).shape[1]

# Separate train-val split
data_train, data_val, chunks_train, chunks_val = train_test_split_chunks(data)

# Calculate mean and standard deviation to normalize the data
train_mean, train_std = compute_center(data_train)

# Format pytorch dataset for the data loader
# Normalize training and validation data according to the training center
train_ds = gpp_dataset(data_train, train_mean, train_std)
val_ds = gpp_dataset(data_val, train_mean, train_std)

for i in tqdm(range(args.num_trials)):
    batch_size = int(np.random.choice(batch_sizes_list))
    hidden_dim = np.random.choice(hidden_dim_list)
    lr = np.random.choice(learning_rates_list)

    print(f"Trial {i+1}/{args.num_trials} | Batch Size: {batch_size} | Hidden Units: {hidden_dim} | Learning Rate: {lr}")

    # Initialize the model
    model = Model(input_dim=INPUT_FEATURES, hidden_dim=hidden_dim).to(device = args.device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False)

    # Call the train_model function with the current set of hyperparameters
    val_r2, val_rmse, model = train_model(train_dl, val_dl, model, optimizer, writer, args.n_epochs, args.device, args.patience)
    
    print(f"R2 Score: {val_r2:.4f} | RMSE: {val_rmse:.4f}")

    # Update best model if current model is better
    if val_rmse < best_validation_score:
        best_validation_score = val_rmse
        best_hyperparameters = {'batch_size': batch_size, 'hidden_units': hidden_dim, 'learning_rate': lr, 'validation_rmse': val_rmse, 'validation_r2': val_r2}
        best_model = model

# Close Tensorboard writer
writer.close()

# Get the current date and time for the filename
current_datetime = datetime.datetime.now().strftime("%d%m%Y_%H%M")
output_file_name = f"best_dnn_model_{current_datetime}"

# Save the best model and hyperparameters
torch.save(best_model.state_dict(), f"../models/{output_file_name}.pt")
with open(f"../models/{output_file_name}_hyperparameters.txt", 'w') as f:
    f.write(str(best_hyperparameters))

print("Best DNN Model Hyperparameters:", best_hyperparameters)