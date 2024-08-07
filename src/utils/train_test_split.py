# This file contains the cross validation data split
# functions used in the main scripts

# Import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_sites(df, test_size=0.2):
    """
    Function to split the DataFrame into train and validation sets
    based on 'TA_F_MDS' mean temperature and 'ai' aridity per 'sitename'.
    20% of data is used for validation.

    Parameters:
    - df (pd.DataFrame): DataFrame containing flux variables used for training and 'TA_F_MDS' and 'sitename' columns

    Returns:
    - df_train (pd.DataFrame): DataFrame with the train data
    - df_val (pd.DataFrame): DataFrame with the validation data
    """

    grouped = add_stratification_target(df)
    
    # Making sure that there is more than one chunk in each combined_target group,
    # in order to run train_test_split.
    special_sites = grouped.groupby('stratify').filter(lambda x: len(x) == 1).index.tolist()
    
    if len(special_sites) == len(grouped.index.unique()):
        # If there are few sites for training (e.g. in leave-vegetation-out)
        # it may lead to all sites being "special", then split at random
        train_df, val_df = train_test_split(grouped, test_size=test_size)

        # Get train and validation sites
        sites_train = train_df.index
        sites_val = val_df.index

    elif len(special_sites):
        # If there is only one site per category, put it in the training set and split the rest
        # stratified by mean temperature and aridity.
        grouped = grouped.drop(special_sites)
        train_df, val_df = train_test_split(grouped, test_size=test_size, stratify=grouped['stratify'])

        # Get train and validation sites
        sites_train = train_df.index.tolist() + special_sites    # Add weird sites to training set
        sites_val = val_df.index 
    
    else:
        # Use train_test_split to create two site groups, stratified by mean temperature and aridity
        train_df, val_df = train_test_split(grouped, test_size=test_size, stratify=grouped['stratify'])

        # Get train and validation sites
        sites_train = train_df.index
        sites_val = val_df.index

    # Separate the time series data
    df_train = df.loc[[any(site == s for s in sites_train) for site in df.index]]
    df_val = df.loc[[any(site == s for s in sites_val) for site in df.index]]

    return df_train, df_val, sites_train, sites_val


def add_stratification_target(df):
    # Group by site and calculate mean temperature
    grouped = df.groupby('sitename').agg({'TA_F_MDS': 'mean', 'ai': 'first'})

    # Discretize numerical columns into bins
    grouped['TA_F_MDS_bins'] = pd.qcut(grouped['TA_F_MDS'], 2, labels=False)
    grouped['ai_bins'] = pd.qcut(grouped['ai'], 2, labels=False)

    # Combine discretized columns into a single categorical column for stratification
    grouped['stratify'] = grouped['TA_F_MDS_bins'].astype(str) + '_' + grouped['ai_bins'].astype(str)

    grouped = grouped.drop(columns=['TA_F_MDS_bins', 'ai_bins'])

    return grouped