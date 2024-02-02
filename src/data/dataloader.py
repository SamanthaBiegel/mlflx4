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
        
        # Define mask for imputed values
        self.mask = ~x['imputed'].values

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
            A vector with the mask for imputed values is also returned.
        """
        
        # Select rows corresponding to site idx
        rows = [s == self.sites[idx] for s in self.sitename]
        return self.x[rows], self.y[rows], self.c[rows], self.mask[rows]
  
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
    x_num = x_num.drop(columns = ['GPP_NT_VUT_REF', 'ai', "chunk_id"])

    # Calculate mean and standard deviation, per column
    x_mean = x_num.mean()
    x_std = x_num.std()

    return x_mean, x_std



class gpp_dataset(Dataset):
    def __init__(self, x, train_mean, train_std, test=False):
        """
        A PyTorch Dataset for GPP prediction, without categorical features.

        Args:
            x (DataFrame): Input data containing numerical features and target variable.
            train_mean (float): Mean value of training data for centering features.
            train_std (float): Standard deviation of training data for scaling features.
        """
        self.test = test
        
        # Select numeric variables only, without GPP
        x_num = x.select_dtypes(include = ['int', 'float'])
        x_num = x_num.drop(columns = ['GPP_NT_VUT_REF', 'ai',"chunk_id"])

        # Center data, according to training data center
        x_centered = (x_num - train_mean)/train_std

        # Create tensor for the covariates
        # The pandas DataFrame must be converted to a numpy array
        self.x = torch.tensor(x_centered.values,
                              dtype = torch.float32)
        
        # Define target        
        self.y = torch.tensor(x['GPP_NT_VUT_REF'].values,
                              dtype = torch.float32)
        
        # Define mask for imputed values
        self.mask = torch.tensor(~x['imputed'].values, dtype = torch.bool)

        if not self.test:
            self.chunks = x["chunk_id"].unique()

            # For each chunk, store the corresponding rows in x
            chunk_to_row = {}
            for chunk in self.chunks:
                chunk_to_row[chunk] = list(x["chunk_id"] == chunk)

            # Reshape x, y and mask to index by chunk
            x_final = torch.zeros((len(self.chunks), 5*365, x_centered.shape[1]))
            y_final = torch.zeros((len(self.chunks), 5*365))
            mask_final = torch.zeros((len(self.chunks), 5*365))
            for i, chunk in enumerate(self.chunks):
                x_final[i,:,:] = self.x[chunk_to_row[chunk],:]
                y_final[i,:] = self.y[chunk_to_row[chunk]]
                mask_final[i,:] = self.mask[chunk_to_row[chunk]]

            self.x = x_final
            self.y = y_final
            self.mask = mask_final.bool()

            # Define length of dataset
            self.len = len(x["chunk_id"].unique())      # number of chunks
        else:
            self.sitename = x.index
            self.sites = x.index.unique()
            self.len = len(self.sites)

    def __getitem__(self, idx):
        """
        Get the covariates and target variable for a specific site.

        Args:
            idx (int): Index of the site.

        Returns:
            Tuple of numerical covariates and target variable for the specified site.
            A vector with the mask for imputed values is also returned.
        """
        if not self.test:
            return self.x[idx,:,:], self.y[idx,:], self.mask[idx,:]
        else:
            rows = [s == self.sites[idx] for s in self.sitename]
            return self.x[rows], self.y[rows], self.mask[rows]

  
    def __len__(self):
        """
        Get the total number of samples (i.e. sites) in the dataset.

        Returns:
            int: The number of samples in the dataset, that is, the number of sites.
        """

        return self.len
