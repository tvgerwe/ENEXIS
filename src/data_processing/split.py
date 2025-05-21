# src/data_processing/split.py

import pandas as pd
from typing import Tuple

def time_based_split(df: pd.DataFrame, time_column: str = "datetime", train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(by=time_column)
    split_index = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()
    return train_df, test_df

def validate_forecast_range(train_df: pd.DataFrame, test_df: pd.DataFrame, time_column: str = "datetime") -> bool:
    train_end = train_df[time_column].max()
    test_start = test_df[time_column].min()
    return train_end < test_start
