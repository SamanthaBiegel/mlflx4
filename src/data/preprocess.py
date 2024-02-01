# Functions used to preprocess data

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def separate_veg_type(df, v, veg_types):
    """
    A function to select the most common vegetation types (given by 'classid' in the GPP dataset)
    and extract one of the types for model training.

    Args:
        df (DataFrame): Input data containing numerical features and target variable.
        v (str): Vegetation type used for training. It can be included in veg_types or not.
        veg_types (str): Vector of vegetation types used for testing.
    """

    # Select data for training
    df_v = df.loc[df['classid'] == v]

    # Select data for testing, with all selected vegetation types except v
    veg_types.remove(v)
    df_veg_types = df.loc[[any(classid == v for v in veg_types) for classid in df['classid']]]

    return df_v, df_veg_types