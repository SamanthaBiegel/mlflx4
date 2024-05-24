# This script loads the raw dataset, removes unnecessary columns, adds site metadata (aridity index),
# imputes missing values using K-nearest neighbors
# for specific columns, and then saves the cleaned and imputed dataset to a new CSV file named 'df_imputed.csv'.

# Load dependencies
import argparse
import os
import glob
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

# Parse arguments 
parser = argparse.ArgumentParser(description='Data pre-processing')

parser.add_argument('-d', '--data_path', type=str,
                    help='Path to the folder containing the FluxDataKit data (in csv format)')

args = parser.parse_args()

# Load data
csv_files = glob.glob(os.path.join(args.data_path, "*_DD_*.csv"))
dataframes = []
for file in csv_files:
    sitename = os.path.basename(file)[4:10]
    df_tmp = pd.read_csv(file, parse_dates=['TIMESTAMP'])
    df_tmp["sitename"] = sitename
    dataframes.append(df_tmp)
df = pd.concat(dataframes, ignore_index=True)

# Add co2 data
df["year"] = df["TIMESTAMP"].dt.year
df["month"] = df["TIMESTAMP"].dt.month
df_co2 = pd.read_csv(
    'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv', 
    skiprows=58, 
    usecols=[0, 1, 3], 
    names=['year', 'month', 'co2_mlo'],
    parse_dates={'date': ['year', 'month']},
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m')
)
df = pd.merge(df, df_co2[['date', 'co2_mlo']], left_on=[df["TIMESTAMP"].dt.to_period('M')], right_on=[df_co2["date"].dt.to_period('M')], how='left').drop(columns=['key_0', 'date'])

# Filter out sites with cropland and wetland land use type
sites_meta = pd.read_csv("../data/external/sites_meta.csv")
sel_sites_vegtype = sites_meta[~sites_meta["igbp_land_use"].isin(["CRO", "WET"])]["sitename"].tolist()
df = df[df.sitename.isin(sel_sites_vegtype)]
print("Nr sites filtered out due to land use: ", len(sites_meta) - len(sel_sites_vegtype))

# Filter out sites with less than 5 years of GPP data
sites_ys = pd.read_csv("../data/external/fdk_site_fullyearsequence.csv", parse_dates=["start_gpp", "end_gpp"], index_col=0)
sites_ys['year_end_gpp'] = sites_ys['end_gpp'].apply(lambda x: x.year if x.month == 12 and x.day >= 30 else x.year - 1)
sites_ys['nyears_gpp'] = sites_ys['year_end_gpp'] - sites_ys['year_start_gpp'] + 1
sites_ys["date_start_gpp"] = pd.to_datetime(sites_ys["year_start_gpp"], format='%Y')
sites_ys["date_end_gpp"] = pd.to_datetime(sites_ys["year_end_gpp"] + 1, format='%Y')
minimum_nr_years = 5
merged_df = df.merge(sites_ys[['sitename', 'date_start_gpp', 'date_end_gpp', 'nyears_gpp']], on='sitename', how='left')
filtered_df = merged_df[(merged_df['TIMESTAMP'] >= merged_df['date_start_gpp']) & (merged_df['TIMESTAMP'] < merged_df['date_end_gpp']) & (merged_df['nyears_gpp'] >= minimum_nr_years)]
filtered_df = filtered_df.drop(columns=['date_start_gpp', 'date_end_gpp', 'nyears_gpp'])
print("Nr sites filtered out due to insufficient years of GPP data: ", len(merged_df.sitename.unique()) - len(filtered_df.sitename.unique()))
df = filtered_df.copy()

# Filter out sites with missing GPP data
def filter_sites_with_nans_gpp(df):
    df_site_nan_gpp = df.groupby('sitename').apply(lambda x: x['GPP_NT_VUT_REF'].isna().all())
    sites_nan_gpp = df_site_nan_gpp[df_site_nan_gpp].index
    df_filtered = df[~df.sitename.isin(sites_nan_gpp)]
    print("Nr sites filtered out due to missing GPP_NT_VUT_REF: ", len(df.sitename.unique()) - len(df_filtered.sitename.unique()))
    return df_filtered
df = filter_sites_with_nans_gpp(df)

# Filter out invalid years
valid_years = pd.read_csv("../data/external/valid_years_final.csv")
sites_df = df.sitename.unique()
sites_vy = valid_years.Site.unique()
diff = np.setdiff1d(sites_df, sites_vy)
print('Sites without valid years information: ', diff)
valid_years['years'] = valid_years['end_year'] - valid_years['start_year'] + 1
valid_years['start_date'] = pd.to_datetime(valid_years['start_year'], format='%Y')
valid_years['end_date'] = pd.to_datetime(valid_years['end_year'] + 1, format='%Y')
merged_df = df.merge(valid_years[['Site', 'start_date', 'end_date', 'years']], left_on='sitename', right_on='Site', how='left')
merged_df = merged_df.fillna({'start_date': pd.to_datetime('1900-01-01'), 'end_date': pd.to_datetime('2100-01-01'), 'years': 5})
filtered_df = merged_df[(merged_df['TIMESTAMP'] >= merged_df['start_date']) & (merged_df['TIMESTAMP'] < merged_df['end_date']) & (merged_df['years'] >= 5)]
filtered_df = filtered_df.drop(columns=['start_date', 'end_date', 'Site'])
print("Nr sites filtered out due to manual evaluation of valid years: ", len(merged_df.sitename.unique()) - len(filtered_df.sitename.unique()))
df = filtered_df.copy()

# Check start and end dates per site
def test_start_end_date(df):
    invalid_dates = []
    for site, group in df.groupby('sitename'):
        start_date = group['TIMESTAMP'].min()
        end_date = group['TIMESTAMP'].max()
        if start_date.day != 1 or start_date.month != 1 or (end_date.day != 31 and end_date != 30) or end_date.month != 12:
            invalid_dates.append(site)
    print("Sites with invalid start/end dates: ", invalid_dates)
test_start_end_date(df)

# Remove years with long gaps
def remove_years_with_long_gaps(df, site_column='sitename', year_column='year', target_column='GPP_NT_VUT_REF', max_gap_length=25):
    to_drop = []
    for (site, year), group in df.groupby([site_column, year_column]):
        nan_sequences = group[target_column].isna().astype(int).groupby(group[target_column].notna().astype(int).cumsum()).sum()
        if nan_sequences.max() > max_gap_length:
            to_drop.append((site, year))
    if to_drop:
        for site, year in to_drop:
            df = df[~((df[site_column] == site) & (df[year_column] == year))]
    print("Nr years filtered out due to long gaps: ", len(to_drop))
    return df
df = remove_years_with_long_gaps(df)

# Filter out sites with less than 5 years of valid GPP data
def filters_sites_with_nans(df):
    """ Filter out sites with less than 5 years of valid GPP data """
    sites_to_remove = []
    for site, group in df.groupby('sitename'):
        df_gpp = group.dropna(subset=['GPP_NT_VUT_REF'])
        max_timestamp = df_gpp['TIMESTAMP'].max()
        min_timestamp = df_gpp['TIMESTAMP'].min()
        if (max_timestamp.year - min_timestamp.year) < 4:
            sites_to_remove.append(site)
    print("Nr sites filtered out due to less than 5 years of valid GPP data: ", len(sites_to_remove))
    return df[~df.sitename.isin(sites_to_remove)]
df = filters_sites_with_nans(df)

# Filter out sites with gaps in timeseries
def test_gaps_within_timeseries(df, date_column="TIMESTAMP", site_column="sitename"):
    gaps = {}
    df[date_column] = pd.to_datetime(df[date_column])
    for site, group in df.groupby(site_column):
        group = group.sort_values(by=date_column).dropna(subset=["GPP_NT_VUT_REF"])
        if any(group[date_column].diff().dt.days > 1):
            gaps[site] = True
    return gaps
sites_gaps = test_gaps_within_timeseries(df)
def filter_sites_with_gaps(df, gaps):
    return df[~df.sitename.isin(gaps.keys())]
df = filter_sites_with_gaps(df, sites_gaps)
print("Nr sites filtered out due to gaps in timeseries: ", len(sites_gaps))

# Impute missing net radiation values
def fill_netrad(df_site):
    if df_site['NETRAD'].isna().sum() > 0 and df_site['NETRAD'].isna().mean() < 0.4:
        kfFEC = 2.04
        df_site.loc[:, "ppfd"] = df_site.loc[:, "SW_IN_F_MDS"] * kfFEC * 1e-6 * 86400

        X = df_site[['TA_DAY_F_MDS', 'ppfd']]
        y = df_site[['NETRAD']]

        scaler = StandardScaler()
        knn_imputer = KNNImputer(n_neighbors=5)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, ['TA_DAY_F_MDS', 'ppfd'])
            ],
            remainder='passthrough'
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
        ])
        X_transformed = pipeline.fit_transform(X)

        complete_data = pd.DataFrame(X_transformed, columns=['TA_DAY_F_MDS', 'ppfd'], index=X.index)
        complete_data['NETRAD'] = df_site['NETRAD']

        netrad_imputed = knn_imputer.fit_transform(complete_data)
        netrad_imputed_df = pd.DataFrame(netrad_imputed, columns=['TA_DAY_F_MDS', 'ppfd', 'NETRAD'], index=complete_data.index)

        return netrad_imputed_df['NETRAD']

    else:
        return df_site['NETRAD']
for site in df['sitename'].unique():
    site_indices = df[df['sitename'] == site].index
    imputed_values = fill_netrad(df.loc[df['sitename'] == site].copy())
    df.loc[site_indices, "NETRAD"] = imputed_values.loc[site_indices]

# Impute missing values for VPD, wind speed, air pressure and daytime temperature
df.loc[df["PA_F"] <= 0, "PA_F"] = np.nan
for column in ["WS_F", "VPD_F_MDS", "PA_F"]:
    for site in df['sitename'].unique():
        mean_value = df.loc[df['sitename'] == site, column].mean()
        df.loc[(df['sitename'] == site) & df[column].isna(), column] = mean_value
df_tmp = df[['TA_F_MDS','SW_IN_F_MDS','TA_DAY_F_MDS', 'sitename']].set_index('sitename')
for s in df["sitename"].unique():
    impute = KNNImputer()
    x = df_tmp[df_tmp.index == s].values
    x = impute.fit_transform(x)
    df.loc[df['sitename'] == s, 'TA_DAY_F_MDS'] = x[:,2]

# Compute aridity index
C_p = 1.013e-3
L = 2.45
df['Delta'] = 4098 * (0.6108 * np.exp((17.27 * df['TA_F_MDS']) / (df['TA_F_MDS'] + 237.3))) / (df['TA_F_MDS'] + 237.3)**2
df['Gamma'] = (C_p * df['PA_F']) / (0.622 * L)
df['PET'] = (0.408 * df['Delta'] * (df['NETRAD']*86400*1e-6) + \
                    (df['Gamma'] * (900 / (df['TA_F_MDS'] + 273)) * df['WS_F'] * (df['VPD_F_MDS']/10))) / \
                    (df['Delta'] + df['Gamma'] * (1 + 0.34 * df['WS_F']))
site_totals = df.dropna(subset=["PET"]).groupby('sitename').agg({'PET': 'sum', 'P_F': 'sum'})
site_totals['ai'] = site_totals['P_F'] / site_totals['PET']
df = pd.merge(df, site_totals[['ai']], on='sitename', how='left')
df['ai'] = df['ai'].fillna(df['ai'].mean())

df_ml = df[["TIMESTAMP", "TA_F_MDS", "TA_DAY_F_MDS", "SW_IN_F_MDS", "LW_IN_F_MDS", "VPD_F_MDS", "PA_F", "P_F", "WS_F", "FPAR", "co2_mlo", "ai", "GPP_NT_VUT_REF", "sitename"]].copy()
df_ml.loc[:, 'chunk_id'] = np.nan

print("Total number of sites: ", len(df_ml.sitename.unique()))

df_ml.to_csv("../data/processed/fdk_v3_ml.csv", index=False)

