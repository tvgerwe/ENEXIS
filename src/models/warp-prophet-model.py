# env - enexis-may-03-env-run
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import polars as pl
import os
from datetime import datetime
import time
import sqlite3
from pathlib import Path
import logging
import json
import joblib
import itertools
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.utils.build_training_set import build_training_set


logging.basicConfig(
    level=logging.INFO,
    filename=str(Path(__file__).parent / "logs" / "warp-prophet-model-json.log"),
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('warp-prophet-model')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.json"

if not CONFIG_PATH.exists():
    logger.error(f"❌ Config not found at : {CONFIG_PATH}")
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DB = PROJECT_ROOT / "src" / "data" / "logs.db"
TABLE_NAME = "prophet_logs"

# Ensure the log database and table exist
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
    check_if_order_evaluated(order, seasonal_order)
    conn = sqlite3.connect(LOG_DB)
    conn.execute(
        f"INSERT INTO {TABLE_NAME} (order_params, seasonal_order_params, rmse) VALUES (?, ?, ?)",
        (str(order), str(seasonal_order), rmse)
    )
    conn.commit()
    conn.close()


# Custom function for MAPE and sMAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

def compute_aic(y_true, y_pred, num_params):
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    n = len(y_true)
    aic = n * np.log(mse) + 2 * num_params
    return aic

results = []

WARP_DATA_FILE_PATH = PROJECT_ROOT / "src" / "data" / "warp-csv-dataset.csv"
logger.info(f"WARP_DATA_FILE_PATH: {WARP_DATA_FILE_PATH}")


try:
    with open(WARP_DATA_FILE_PATH, 'rb') as csv_file:
        df_pd_orig = pd.read_csv(csv_file)
    logger.info(f"Loaded data from {WARP_DATA_FILE_PATH}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

df_pd_orig['datetime'] = df_pd_orig['target_datetime']
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
df = df_pd_orig.sort_values(by='datetime')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)

y = df[['datetime', 'Price']]
X = df.drop(columns=['Price'])

# Ensure train/test splits use the same mask for X and y to avoid length mismatch
train_start = "2025-01-01 00:00:00"
train_end = "2025-03-14 23:00:00"
test_start = "2025-03-15 00:00:00"
test_end = "2025-03-22 00:00:00"

mask_train = (df['datetime'] >= train_start) & (df['datetime'] <= train_end)
mask_test = (df['datetime'] >= test_start) & (df['datetime'] <= test_end)

X_train = X[mask_train].copy().reset_index(drop=True)
y_train = y[mask_train].copy().reset_index(drop=True)
X_test = X[mask_test].copy().reset_index(drop=True)
y_test = y[mask_test].copy().reset_index(drop=True)

logger.info(f"Train Date Range: Start: {X_train['datetime'].min()} End: {X_train['datetime'].max()}")
logger.info(f"Test Date Range: Start: {X_test['datetime'].min()} End: {X_test['datetime'].max()}")

regressors = [
    'Load','shortwave_radiation','temperature_2m','direct_normal_irradiance','diffuse_radiation','Flow_NO','yearday_cos','Flow_GB','month',
    'is_dst','yearday_sin','is_non_working_day','hour_cos','is_weekend','cloud_cover','weekday_sin','hour_sin','weekday_cos'
]

# removed wind_speed_10m as NaN values were causing issues

available_regressors = [col for col in regressors if col in X_train.columns]

train_prophet = pd.concat([
    y_train[['datetime', 'Price']],
    X_train[available_regressors]
], axis=1).reset_index(drop=True)
test_prophet = pd.concat([
    y_test[['datetime', 'Price']],
    X_test[available_regressors]
], axis=1).reset_index(drop=True)

train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)

train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

logger.info("Prophet Training Data Sample:")
logger.info(f"\n{train_prophet.head()}")
logger.info(f"y_train columns: {y_train.columns.tolist()}")
logger.info(f"X_train columns (used regressors): {available_regressors}")

# --- Hyperparameter tuning for Prophet ---
param_grid = {
    'changepoint_prior_scale': [0.5],
    'seasonality_mode': ['additive'],
    'seasonality_prior_scale': [1.0],
    'holidays_prior_scale': [1.0],    
    'changepoint_range': [0.8]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
best_params = None
best_mae = float('inf')
best_model = None
best_forecast = None

model_run_start_time = time.time()
for params in all_params:
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        seasonality_prior_scale=params['seasonality_prior_scale']
    )
    for reg in available_regressors:
        model.add_regressor(reg)
    model.fit(train_prophet)
    forecast = model.predict(test_prophet)
    y_true = test_prophet['y'].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    logger.info(f"Params: {params} → MAE: {mae:.2f}")
    if mae < best_mae:
        best_mae = mae
        best_params = params
        best_model = model
        best_forecast = forecast

logger.info(f"Best Parameters Found: {best_params}")
logger.info(f"Best MAE: {best_mae:.2f}")

model = best_model
forecast = best_forecast
model_run_end_time = time.time()
logger.info("✅ Forecast complete")

forecast_indexed = forecast.set_index('ds')
test_prophet_indexed = test_prophet.set_index('ds')
merged = test_prophet_indexed[['y']].join(forecast_indexed[['yhat']], how='inner').dropna()

logger.info(f"Aligned data length: {len(merged)}")
logger.info(f"Date range: {merged.index.min()} to {merged.index.max()}")

diff = merged['y'] - merged['yhat']

mae = mean_absolute_error(merged['y'], merged['yhat'])
mse = mean_squared_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mse)
r2 = r2_score(merged['y'], merged['yhat'])

logger.info(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

model.plot(forecast)
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.show()

model.plot_components(forecast)
plt.tight_layout()
plt.show()

y_true = merged['y'].values
y_pred = merged['yhat'].values
execution_time = model_run_end_time - model_run_start_time
logger.info(f"Execution time: {execution_time:.2f} seconds")

logger.info("\n📊 Evaluation Metrics:")
logger.info(f"Model Name: Prophet")
logger.info(f"MAE: {mae:.3f}")
logger.info(f"MSE: {mse:.3f}")
logger.info(f"RMSE: {rmse:.3f}")
logger.info(f"R²: {r2:.3f}")



comments = "run on 26th May refresh data"
log_model_result(str(best_params), str(regressors), rmse)

model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", mae, mse, rmse, r2, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv"

try:
    if os.path.exists(model_results_file_path):
        existing_results = pd.read_csv(model_results_file_path)
        updated_results = pd.concat([existing_results, metrics_df], ignore_index=True)
        updated_results.to_csv(model_results_file_path, index=False)
        logger.info(f"Appended and saved to {model_results_file_path}")
    else:
        metrics_df.to_csv(model_results_file_path, index=False)
        logger.info(f"Created new CSV at {model_results_file_path}")
except Exception as e:
    logger.error(f"CSV save failed: {e}")
    raise

model_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "prophet_hyper_tuned_model.pkl"
try:
    joblib.dump(model, model_file_path)
    logger.info(f"Model saved at {model_file_path}")
except Exception as e:
    logger.error(f"Model save failed: {e}")
    raise
logger.info(f"✅ Model evaluation saved to {model_results_file_path}")