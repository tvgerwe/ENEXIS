# src/models/naive_model.py

import pandas as pd
from sklearn.metrics import root_mean_squared_error

def run_naive_model(y_true: pd.Series, lag: int = 168) -> pd.Series:
    if len(y_true) <= lag:
        raise ValueError(f"Test set is too small for lag {lag}. Minimum required rows: {lag + 1}, found: {len(y_true)}")
    return y_true.shift(lag)

def evaluate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = ~y_pred.isna()
    common_index = y_true.index.intersection(y_pred[mask].index)
    return root_mean_squared_error(y_true.loc[common_index], y_pred.loc[common_index])