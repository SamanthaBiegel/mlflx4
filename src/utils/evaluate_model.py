from sklearn.metrics import root_mean_squared_error
import numpy as np

def evaluate_model(val_dl, y_pred):
    data_val_eval = val_dl.dataset.data.copy()

    data_val_eval['gpp_pred'] = [item for sublist in y_pred for item in sublist]

    nan_y_true = data_val_eval["GPP_NT_VUT_REF"].isna()
    nan_y_pred = data_val_eval["gpp_pred"].isna()

    data_val_eval = data_val_eval[~(nan_y_true | nan_y_pred)]

    return compute_metrics(data_val_eval["GPP_NT_VUT_REF"], data_val_eval["gpp_pred"]), data_val_eval


def compute_metrics(y_true, y_pred):
    """
    Compute R2 and RMSE metrics
    """
    r2 = y_true.corr(y_pred) ** 2
    rmse = root_mean_squared_error(y_true, y_pred)
    nmae = np.mean(np.abs(y_true - y_pred)) / np.mean(y_true)
    abs_bias = np.abs(np.mean(y_true) - np.mean(y_pred))
    
    return {"r2": r2, "rmse": rmse, "nmae": nmae, "abs_bias": abs_bias}