# This script loads the raw dataset, removes unnecessary columns, adds site metadata (aridity index),
# imputes missing values using K-nearest neighbors
# for specific columns, and then saves the cleaned and imputed dataset to a new CSV file named 'df_imputed.csv'.

# Load dependencies
import argparse
import os
import glob
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

# Parse arguments 
parser = argparse.ArgumentParser(description='Data pre-processing')

parser.add_argument('-d', '--data_path', type=str,
                    help='Path to the folder containing the FluxDataKit data (in csv format)')

args = parser.parse_args()

def chunk_data(df, years=5):

    sites = df.index.unique()

    chunk_size = years * 365

    chunks_per_site = {}
    chunks = []
    chunk_idx = 0

    # Iterate over sites
    for site in sites:
        # Calculate number of chunks for the site
        num_chunks = len(df[df.index == site]) // chunk_size
        chunks_per_site[site] = num_chunks
        site_size = len(df[df.index == site])

        # Create a list of chunk indices for each site
        for i in range(num_chunks):
            chunks.append([chunk_idx] * chunk_size)
            chunk_idx += 1

        # Record the leftover data points
        leftover = site_size % chunk_size
        if leftover > 0:
            chunks.append([np.nan] * leftover)

    flattened_chunks = [item for sublist in chunks for item in sublist]

    # Add the chunk indices to the dataframe
    return df.assign(chunk_id=flattened_chunks)

# Load the raw dataset from the CSV files
# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(args.data_path, "*_DD_*.csv"))

# Initialize an empty list to store the dataframes
dataframes = []

# Iterate over the CSV files
for file in csv_files:
    # Extract the sitename from the filename
    sitename = os.path.basename(file)[4:10]
    
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
    
    # Add the sitename as a new column in the dataframe
    df["sitename"] = sitename

    # Generate the filename for the "HH" file by replacing "DD" with "HH" in the original file path
    file_HH = file.replace("_DD_", "_HH_")
    
    # Check if the "HH" file exists to avoid errors
    if os.path.exists(file_HH):
        # Read the "HH" CSV file into a pandas dataframe
        df_HH = pd.read_csv(file_HH, parse_dates=['TIMESTAMP_START'], usecols=['TIMESTAMP_START', 'TA_F_MDS', 'SW_IN_F_MDS'])
        df_HH["TIMESTAMP"] = df_HH["TIMESTAMP_START"].dt.date

        daytime_thresh = df_HH['SW_IN_F_MDS'].quantile(0.01)

        # Use threshold to get nighttime entries
        df_HH['NIGHT'] = df_HH['SW_IN_F_MDS'] <= daytime_thresh

        # Filter for nighttime and calculate averages by TIMESTAMP for nighttime
        df_night = df_HH[df_HH['NIGHT']].groupby('TIMESTAMP').agg(
            TA_NIGHT_F_MDS=pd.NamedAgg(column='TA_F_MDS', aggfunc=lambda x: x.mean(skipna=True)),
        ).reset_index()

        # parse timestamp to datetime
        df_night['TIMESTAMP'] = pd.to_datetime(df_night['TIMESTAMP'])

        df = pd.merge(df, df_night, on='TIMESTAMP', how='left')
    
    # Append the dataframe to the list
    dataframes.append(df)

# Merge all the dataframes together
merged_df = pd.concat(dataframes, ignore_index=True).set_index("sitename")

# Data cleaning
merged_df = merged_df[merged_df["PA_F"] > 0]
merged_df = merged_df[merged_df["TA_F_MDS"] > -50]
merged_df = merged_df[merged_df["TA_DAY_F_MDS"] > -50]
merged_df = merged_df[merged_df["TA_NIGHT_F_MDS"] > -50]
merged_df = merged_df[merged_df["LW_IN_F_MDS"] < 1000]

# Read metadata from FLUXNET sites to obtain aridity index (ai)
# This file was provided by Beni and includes only 53 sites, in the future it may
# contain more variables to extend the flux data, but the pipeline to obtain
# the site characteristics should be written transparently.
# df_meta = pd.read_csv("../data/external/fluxnet2015_sites_metainfo.csv", index_col = 0)
# df_meta.set_index('mysitename', inplace=True)

# Calculate aridity index using Penman-Monteith equation, since some sites are missing from metadata file
# Constants
C_p = 1.013e-3  # MJ/kg°C, specific heat of air at constant pressure
L = 2.45  # MJ/kg, latent heat of vaporization

# Calculate Δ (Slope of vapor pressure curve)
merged_df['Delta'] = 4098 * (0.6108 * np.exp((17.27 * merged_df['TA_F_MDS']) / (merged_df['TA_F_MDS'] + 237.3))) / (merged_df['TA_F_MDS'] + 237.3)**2

# Calculate γ (Psychrometric constant)
merged_df['Gamma'] = (C_p * merged_df['PA_F']) / (0.622 * L)

# Calculate PET using the Penman-Monteith equation (FAO-56 reference crop evapotranspiration)
merged_df['PET'] = (0.408 * merged_df['Delta'] * (merged_df['NETRAD']*86400*1e-6) + \
                    (merged_df['Gamma'] * (900 / (merged_df['TA_F_MDS'] + 273)) * merged_df['WS_F'] * (merged_df['VPD_F_MDS']/10))) / \
                    (merged_df['Delta'] + merged_df['Gamma'] * (1 + 0.34 * merged_df['WS_F']))

# Calculate aridity index (ai) per site
site_totals = merged_df.dropna(subset=["PET"]).groupby('sitename').agg({'PET': 'sum', 'P_F': 'sum'})
site_totals['ai'] = site_totals['P_F'] / site_totals['PET']

# Merge the data with the metadata based on their indices
# The aridity index will be used for the stratified train-test splits, not for modelling
# data = pd.merge(data, df_meta[['ai']], left_on='sitename', right_index=True, how='left')
merged_df = pd.merge(merged_df, site_totals[['ai']], left_index=True, right_index=True, how='left')

data = merged_df[["TIMESTAMP", "TA_F_MDS", "TA_DAY_F_MDS", "TA_NIGHT_F_MDS", "SW_IN_F_MDS", "LW_IN_F_MDS", "VPD_F_MDS", "PA_F", "P_F", "WS_F", "FPAR", "ai", "GPP_NT_VUT_REF"]].copy()

print("Imputing temperature (day and night) and GPP values")

sites = data.index.unique()
# Impute 'TA_F_DAY' and 'TA_F_NIGHT' columns using 'TA_F' and 'SW_IN_F'
# Iterate over sites to perform imputation for each site
df = data[['TA_F_MDS','SW_IN_F_MDS','TA_DAY_F_MDS', 'TA_NIGHT_F_MDS']]
for s in sites:
    impute = KNNImputer()
    x = df[df.index == s].values
    x = impute.fit_transform(x)
    data.loc[data.index == s, 'TA_DAY_F_MDS'] = x[:,2]
    data.loc[data.index == s, 'TA_NIGHT_F_MDS'] = x[:,3]

# Impute 'GPP_NT_VUT_REF' column using multiple, selected features
# Iterate over sites to perform imputation for each site
df = data[['TA_F_MDS','SW_IN_F_MDS','TA_DAY_F_MDS', 'LW_IN_F_MDS','WS_F','P_F', 'VPD_F_MDS', 'GPP_NT_VUT_REF']]
for s in sites:
    impute = KNNImputer()
    x = df[df.index == s].values
    x = impute.fit_transform(x)
    data.loc[data.index == s, 'GPP_NT_VUT_REF'] = x[:,-1]

# Add a column indicating whether the GPP values were imputed (True) or original (False)
# to be used as a mask in the model testing
data.loc[:, 'imputed'] = df.loc[:, 'GPP_NT_VUT_REF'].isna()

# Fill in missing wind speed or VPD with mean values
for site in data[data["WS_F"].isna()].index.unique():
    data.loc[data.index == site, 'WS_F'] = data.loc[data.index == site, 'WS_F'].mean()
for site in data[data["VPD_F_MDS"].isna()].index.unique():
    data.loc[data.index == site, 'VPD_F_MDS'] = data.loc[data.index == site, 'VPD_F_MDS'].mean()

# Remove sites with missing FPAR
data = data[data.index != 'US-BZS']
data = data[data.index != 'US-ORv']

data = chunk_data(data)

# Filter data to remove sites with less than 5 years of data
lengths = data.groupby(data.index).size()/365
lengths = lengths[lengths > 5]
data = data[data.index.isin(lengths.index)]

# Save the cleaned and imputed dataset to a new CSV file    
data.to_csv('../data/processed/df_imputed.csv')
print("Imputed data saved to data/processed/df_imputed.csv")