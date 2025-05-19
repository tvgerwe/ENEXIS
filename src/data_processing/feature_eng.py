# src/data_processing/feature_eng.py
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_lag_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, columns: list, windows: list) -> pd.DataFrame:
    for col in columns:
        for window in windows:
            df[f"{col}_rollmean_{window}"] = df[col].rolling(window).mean()
            df[f"{col}_rollstd_{window}"] = df[col].rolling(window).std()
    return df

def add_time_features(df: pd.DataFrame, time_column: str = "datetime") -> pd.DataFrame:
    df[time_column] = pd.to_datetime(df[time_column])
    df["hour"] = df[time_column].dt.hour
    df["dayofweek"] = df[time_column].dt.dayofweek
    df["month"] = df[time_column].dt.month
    df["is_weekend"] = df["dayofweek"] >= 5
    return df

def scale_features(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled, scaler
