import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_naive_model(df: pd.DataFrame, target_col="Price", shift_hours=24):
    df = df.copy()
    df["prediction"] = df[target_col].shift(shift_hours)
    df = df.dropna()

    y_true = df[target_col]
    y_pred = df["prediction"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"🧠 Naive model ({shift_hours}h shift)")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return df[["prediction"]], {"mae": mae, "rmse": rmse}
