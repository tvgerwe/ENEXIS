import pandas as pd
import numpy as np
import holidays

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["date"] = df["datetime"].dt.date

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Weekend/holiday flags
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    nl_holidays = holidays.country_holidays("NL", years=df["datetime"].dt.year.unique().tolist())
    df["is_holiday"] = df["date"].isin(nl_holidays).astype(int)
    df["is_non_working_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)

    return df
