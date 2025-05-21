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

logging.basicConfig(
    level=logging.INFO,
    filename=str(Path(__file__).parent / "warp-prophet-model-json.log"),
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('warp-prophet-model')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

if not CONFIG_PATH.exists():
    logger.error(f"❌ Config not found at : {CONFIG_PATH}")
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_RUN_RESULTS_DIR = config['ned']['ned_model_download_dir']
logger.info(f"MODEL_RUN_RESULTS_DIR: {MODEL_RUN_RESULTS_DIR}")

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

CSV_DATA_DIR = config['ned']['ned_model_download_dir']
csv_file_path = os.path.join(CSV_DATA_DIR, f"warp-csv-dataset.csv")

try:
    with open(csv_file_path, 'rb') as csv_file:
        df_pd_orig = pd.read_csv(csv_file)
    logger.info(f"Loaded data from {csv_file_path}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

df_pd_orig['datetime'] = df_pd_orig['target_datetime']
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
df = df_pd_orig.sort_values(by='datetime')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)

y = df[['datetime', 'Price']]
X = df.drop(columns=['Price'])

train_start = "2025-01-01"
train_end   = "2025-04-30"
test_start  = "2025-05-01"
test_end    = "2025-05-19"

X_train = X[(X['datetime'] >= train_start) & (X['datetime'] <= train_end)].copy()
X_test  = X[(X['datetime'] >= test_start) & (X['datetime'] <= test_end)].copy()
y_train = y[(y['datetime'] >= train_start) & (y['datetime'] <= train_end)].copy()
y_test  = y[(y['datetime'] >= test_start) & (y['datetime'] <= test_end)].copy()

logger.info(f"Train Date Range: Start: {X_train['datetime'].min()} End: {X_train['datetime'].max()}")
logger.info(f"Test Date Range: Start: {X_test['datetime'].min()} End: {X_test['datetime'].max()}")

regressors = ['month','shortwave_radiation','apparent_temperature','temperature_2m','direct_normal_irradiance','diffuse_radiation','yearday_sin',
              'Flow_BE','hour_sin','is_non_working_day','is_weekend','is_holiday','weekday_cos','wind_speed_10m','hour_cos','weekday_sin',
              'cloud_cover','Flow_GB','Nuclear_Vol','yearday_cos','Flow_NO','Load']

available_regressors = [col for col in regressors if col in X_train.columns]

train_prophet = pd.concat([
    y_train[['datetime', 'Price']].reset_index(drop=True),
    X_train[available_regressors].reset_index(drop=True)
], axis=1)
test_prophet = pd.concat([
    y_test[['datetime', 'Price']].reset_index(drop=True),
    X_test[available_regressors].reset_index(drop=True)
], axis=1)

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
    'changepoint_prior_scale': [0.1],
    'seasonality_mode': ['additive'],
    'seasonality_prior_scale': [1.0]
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

comments = "run on 21st May with merge file"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", mae, mse, rmse, r2, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2", "Comments", "Execution Time", "Run At"])

model_results_file_path = f'{MODEL_RUN_RESULTS_DIR}warp-prophet-model-results.csv'
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

model_file_path = f'{MODEL_RUN_RESULTS_DIR}prophet_hyper_tuned_model.pkl'
try:
    joblib.dump(model, model_file_path)
    logger.info(f"Model saved at {model_file_path}")
except Exception as e:
    logger.error(f"Model save failed: {e}")
    raise
logger.info(f"✅ Model evaluation saved to {model_results_file_path}")