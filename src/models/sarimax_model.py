# src/models/sarimax_model.py

import pandas as pd
import sqlite3
from typing import Tuple, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

LOG_DB = "src/data/logs.db"
TABLE_NAME = "model_performance_log"

def check_if_order_evaluated(order: Tuple, seasonal_order: Tuple) -> bool:
    conn = sqlite3.connect(LOG_DB)
    query = f"""
        SELECT COUNT(*) FROM {TABLE_NAME}
        WHERE order_params = ? AND seasonal_order_params = ?
    """
    params = (str(order), str(seasonal_order))
    result = conn.execute(query, params).fetchone()[0]
    conn.close()
    return result > 0

def log_order_result(order: Tuple, seasonal_order: Tuple, rmse: float, mae: float) -> None:
    conn = sqlite3.connect(LOG_DB)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            order_params TEXT,
            seasonal_order_params TEXT,
            rmse REAL,
            mae REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(f"""
        INSERT INTO {TABLE_NAME} (model_name, order_params, seasonal_order_params, rmse, mae)
        VALUES (?, ?, ?, ?, ?)
    """, ("SARIMAX", str(order), str(seasonal_order), rmse, mae))
    conn.commit()
    conn.close()

def run_sarimax(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, order: Tuple, seasonal_order: Tuple) -> Tuple[Optional[pd.Series], dict]:
    if check_if_order_evaluated(order, seasonal_order):
        print(f"✅ Skipped previously evaluated order: {order}, {seasonal_order}")
        return None, {"rmse": None, "mae": None}

    model = SARIMAX(train_df[target_column], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast = results.forecast(steps=len(test_df))
    y_true = test_df[target_column].values
    y_pred = forecast.values

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)

    log_order_result(order, seasonal_order, rmse, mae)

    return pd.Series(y_pred, index=test_df.index), {"rmse": rmse, "mae": mae}

def auto_arima_order(train_df: pd.DataFrame, target_column: str, seasonal: bool = True, m: int = 24) -> Tuple[Tuple, Tuple]:
    model = auto_arima(train_df[target_column], seasonal=seasonal, m=m, suppress_warnings=True, stepwise=True)
    return model.order, model.seasonal_order
