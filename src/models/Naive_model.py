# src/models/naive_model.py

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

LAG = 168  # fixed lag for 1-week-ahead forecast

def naive_forecast(test_df: pd.DataFrame, target_column: str) -> pd.Series:
    return test_df[target_column].shift(LAG)

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = ~y_pred.isna()
    if mask.sum() == 0:
        raise ValueError("No valid predictions to evaluate (all are NaN).")
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    return {
        "rmse": rmse,
        "mae": mae,
        "n": mask.sum()
    }

def run_naive_model(test_df: pd.DataFrame, target_column: str = "price") -> Tuple[pd.Series, dict]:
    if len(test_df) <= LAG:
        raise ValueError(
            f"Test set is too small for fixed lag {LAG}. "
            f"Minimum required rows: {LAG + 1}, found: {len(test_df)}"
        )
    y_pred = naive_forecast(test_df, target_column)
    y_true = test_df[target_column]
    metrics = evaluate_forecast(y_true, y_pred)
    return y_pred, metrics
