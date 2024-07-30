import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class gpp_dataset_cat(Dataset):
    def __init__(self, x, x_cat, train_mean, train_std):
        """
        A PyTorch Dataset for GPP prediction, with categorical features.

        Args:
            x (DataFrame): Input data containing numerical features and target variable.
            x_cat (DateFrame): Input data containing vegetation and land use classes, encoded
                               by dummy variables.
            train_mean (float): Mean value of training data for centering features.
            train_std (float): Standard deviation of training data for scaling features.
        """
        
        # Select numeric variables only, without GPP and aridity index
        x_num = x.select_dtypes(include = ['int', 'float'])
        x_num = x_num.drop(columns = ['GPP_NT_VUT_REF', 'ai'])
        
        # Center data, according to training data center
        x_centered = (x_num - train_mean)/train_std

        # Create tensor for the covariates
        # The pandas DataFrame must be converted to a numpy array
        self.x = torch.tensor(x_centered.values,
                              dtype = torch.float32)
        self.c = torch.tensor(x_cat.values,
                              dtype = torch.float32)
        
        # Define target        
        self.y = torch.tensor(x['GPP_NT_VUT_REF'].values,
                              dtype = torch.float32)

        # Define vector of sites corresponding to the rows in x
        # to be used for indexing
        self.sitename = x.index

        # Define list of unique sites
        self.sites = x.index.unique()

        # Define length of dataset
        # self.len = x.shape[0]         # number of rows
        self.len = len(self.sites)      # number of sites

    def __getitem__(self, idx):
        """
        Get the covariates and target variable for a specific site.

        Args:
            idx (int): Index of the site.

        Returns:
            Thruple of numerical and categorical covariates and target variable for the specified site.
        """
        
        # Select rows corresponding to site idx
        rows = [s == self.sites[idx] for s in self.sitename]
        return self.x[rows], self.y[rows], self.c[rows]
  
    def __len__(self):
        """
        Get the total number of samples (i.e. sites) in the dataset.

        Returns:
            int: The number of samples in the dataset, that is, the number of sites.
        """

        return self.len
    

def compute_center(x):

    # Select numeric variables only, without GPP and aridity index
    x_num = x.select_dtypes(include = ['int', 'float'])
    x_num = x_num.drop(columns = ['GPP_NT_VUT_REF', 'ai'])

    # Calculate mean and standard deviation, per column
    x_mean = x_num.mean()
    x_std = x_num.std()

    return x_mean, x_std


import torch
from torch.utils.data import Dataset
import random

class gpp_dataset(Dataset):
    def __init__(self, x, train_mean, train_std, test=False, chunk_size=128, overlap=32, max_offset=96):
        """
        A PyTorch Dataset for GPP prediction, without categorical features.

        Args:
            x (DataFrame): Input data containing numerical features and target variable.
            train_mean (float): Mean value of training data for centering features.
            train_std (float): Standard deviation of training data for scaling features.
            test (bool): Flag indicating whether the dataset is for testing.
            chunk_size (int): Size of each chunk/window.
            overlap (int): Overlap size between consecutive chunks/windows.
            max_offset (int): Maximum random offset for varying the starting point.
        """
        self.test = test
        self.data = x
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_offset = max_offset
        
        # Select numeric variables only, without GPP
        x_num = x.select_dtypes(include=['int', 'float'])
        x_num = x_num.drop(columns=['GPP_NT_VUT_REF', 'ai'])

        # Center data, according to training data center
        x_centered = (x_num - train_mean) / train_std

        # Create tensor for the covariates
        self.x = torch.tensor(x_centered.values, dtype=torch.float32)
        
        # Define target        
        self.y = torch.tensor(x['GPP_NT_VUT_REF'].values, dtype=torch.float32)

        # Get max number of samples in one site
        self.max_samples = x.index.value_counts().max()
        
        if not self.test:
            self.sites = x.index.unique()
            self.sitename = x.index

            # Prepare lists to store chunks
            x_chunks_list = []
            y_chunks_list = []

            for site in self.sites:
                site_mask = self.sitename == site
                
                x_site = self.x[site_mask]
                y_site = self.y[site_mask]

                # Randomize the start offset
                start_offset = random.randint(0, self.max_offset)
                
                # Create chunks for the site
                x_chunks = self.create_sliding_chunks(x_site, self.chunk_size, self.overlap, start_offset)
                y_chunks = self.create_sliding_chunks(y_site.unsqueeze(1), self.chunk_size, self.overlap, start_offset)
                
                # Append chunks to lists
                x_chunks_list.extend(x_chunks)
                y_chunks_list.extend(y_chunks)

            # Convert lists to tensors
            self.x = torch.stack(x_chunks_list)
            self.y = torch.stack(y_chunks_list).squeeze(-1)
            self.len = len(self.x)
        else:
            self.sitename = x.index
            self.sites = x.index.unique()
            self.len = len(self.sites)

    def create_sliding_chunks(self, sequence, chunk_size, overlap, start_offset=0):
        chunks = []
        step = chunk_size - overlap
        start = start_offset
        for i in range(start, len(sequence), step):
            end = i + chunk_size
            if end <= len(sequence):
                chunks.append(sequence[i:end])
        return chunks

    def __getitem__(self, idx):
        """
        Get the covariates and target variable for a specific chunk.

        Args:
            idx (int): Index of the chunk.

        Returns:
            Tuple of numerical covariates and target variable for the specified chunk.
        """
        if not self.test:
            return self.x[idx], self.y[idx]
        else:
            rows = [s == self.sites[idx] for s in self.sitename]
            return self.x[rows], self.y[rows]

    def __len__(self):
        """
        Get the total number of chunks in the dataset.

        Returns:
            int: The number of chunks in the dataset.
        """
        return self.len
    

class gpp_dataset_onesite(Dataset):
    def __init__(self, x, train_mean, train_std, test=False):
        """
        A PyTorch Dataset for GPP prediction, without categorical features.

        Args:
            x (DataFrame): Input data containing numerical features and target variable.
            train_mean (float): Mean value of training data for centering features.
            train_std (float): Standard deviation of training data for scaling features.
        """
        self.test = test
        self.data = x
        
        # Select numeric variables only, without GPP
        x_num = x.select_dtypes(include = ['int', 'float'])
        x_num = x_num.drop(columns = ['GPP_NT_VUT_REF', 'ai'])

        # Center data, according to training data center
        x_centered = (x_num - train_mean)/train_std

        # Create tensor for the covariates
        # The pandas DataFrame must be converted to a numpy array
        self.x = torch.tensor(x_centered.values,
                              dtype = torch.float32).unsqueeze(0)
        
        # Define target        
        self.y = torch.tensor(x['GPP_NT_VUT_REF'].values,
                              dtype = torch.float32).unsqueeze(0)
                
        self.mask_padding = torch.ones((1, self.x.shape[1]), dtype = torch.bool)

        self.len = 1

    def __getitem__(self, idx):
        """
        Get the covariates and target variable for a specific site.

        Args:
            idx (int): Index of the site.

        Returns:
            Tuple of numerical covariates and target variable for the specified site.
            A vector with the mask for padded values is also returned.
        """
        if not self.test:
            return self.x[idx,:,:], self.y[idx,:], self.mask_padding[idx,:]
        else:
            rows = [s == self.sites[idx] for s in self.sitename]
            return self.x[rows], self.y[rows], self.mask_padding[rows]
  
    def __len__(self):
        """
        Get the total number of samples (i.e. sites) in the dataset.

        Returns:
            int: The number of samples in the dataset, that is, the number of sites.
        """

        return self.len
