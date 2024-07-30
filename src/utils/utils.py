# This file defines general purpose functions

# Import necessary libraries
import numpy as np
import torch
import random
import os
from sklearn.metrics import root_mean_squared_error
import datetime

# Define a function to set random seeds for reproducibility
def set_seed(seed: int = 42):
    # Set random seed for Python's random module
    random.seed(seed)
    
    # Set environment variable for Python's hash seed
    os.environ['PYHTONHASHSEED'] = str(seed)
    
    # Set random seed for NumPy
    np.random.seed(seed)
    
    # Set random seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set random seed for PyTorch on GPU (if available)
    torch.cuda.manual_seed(seed)
    
    # Ensure deterministic behavior for CuDNN (CuDNN is a GPU-accelerated library)
    torch.backends.cudnn.deterministic = True


def generate_filename(model, desc, hparams):
        # For all hparams replace . with p
        hparams_formatted = {k: f"{v}".replace('-', 'm').replace('.', 'p') for k, v in hparams.items()}
        
        # Get current date and time
        current_datetime = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        
        # Generate filename
        filename = (f"{model}_{desc}_"
                        f"{'_'.join([f'{k}{v}' for k, v in hparams_formatted.items()])}"
                        f"_{current_datetime}")
        return f"{filename}"