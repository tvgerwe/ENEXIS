# src/models/sarimax_model.py

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - sarimax - %(levelname)s - %(message)s")
logger = logging.getLogger("sarimax_model")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DB = PROJECT_ROOT / "src" / "data" / "logs.db"
TABLE_NAME = "sarimax_logs"

def ensure_log_db():
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOG_DB)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_params TEXT,
            seasonal_order_params TEXT,
            rmse REAL
        )
    """)
    conn.commit()
    conn.close()

def log_model_result(order: Tuple, seasonal_order: Tuple, rmse: float):
    ensure_log_db()
    conn = sqlite3.connect(LOG_DB)
    conn.execute(
        f"INSERT INTO {TABLE_NAME} (order_params, seasonal_order_params, rmse) VALUES (?, ?, ?)",
        (str(order), str(seasonal_order), rmse)
    )
    conn.commit()
    conn.close()

def run_sarimax(
    train_df: pd.Series,
    X_train: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame],
    order: Tuple,
    seasonal_order: Tuple
) -> Tuple[pd.Series, float]:

    logger.info(f"📈 Fitting SARIMAX with order={order}, seasonal_order={seasonal_order}")
    model = SARIMAX(
        train_df,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=len(X_test), exog=X_test)
    y_pred = forecast.predicted_mean

    try:
        y_true = train_df.iloc[-len(y_pred):]
        rmse = root_mean_squared_error(y_true, y_pred)
        logger.info(f"📊 RMSE: {rmse:.2f}")
        log_model_result(order, seasonal_order, rmse)
    except Exception as e:
        logger.warning(f"⚠️ Geen RMSE gelogd: {e}")
        rmse = None

    return y_pred, rmse
