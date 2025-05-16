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

def check_if_order_evaluated(order: Tuple, seasonal_order: Tuple) -> bool:
    ensure_log_db()
    conn = sqlite3.connect(LOG_DB)
    query = f"""
        SELECT COUNT(*) FROM {TABLE_NAME}
        WHERE order_params = ? AND seasonal_order_params = ?
    """
    params = (str(order), str(seasonal_order))
    result = conn.execute(query, params).fetchone()[0]
    conn.close()
    return result > 0

def log_model_result(order: Tuple, seasonal_order: Tuple, rmse: float):
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
) -> Tuple[Optional[pd.Series], Optional[float]]:

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

    steps = len(X_test) if X_test is not None else 168  # fallback
    forecast = results.get_forecast(steps=steps, exog=X_test)
    y_pred = forecast.predicted_mean

    if X_test is not None:
        y_pred.index = X_test.index
    else:
        start = train_df.index[-1] + pd.Timedelta(hours=1)
        y_pred.index = pd.date_range(start=start, periods=steps, freq="H")

    try:
        y_true = train_df.iloc[-steps:]
        rmse = root_mean_squared_error(y_true, y_pred)
        logger.info(f"📊 RMSE: {rmse:.2f}")
        log_model_result(order, seasonal_order, rmse)
    except Exception as e:
        rmse = None
        logger.warning(f"⚠️ Geen RMSE gelogd: {e}")

    return y_pred, rmse