import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import timedelta
from pathlib import Path
import logging
import json

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(Path(__file__).parent / "logs" / "warp-prophet_forecast_tuned_model-json.log"),
    filemode='a'
)
logger = logging.getLogger('prophet_forecast_tuned_model')


# === Config Setup ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

logger.info("PROJECT_ROOT" + str(PROJECT_ROOT))

model_file_path = PROJECT_ROOT / "sandeep" / "ned" / "downloads" / "prophet_hyper_tuned_model.pkl"
forecast_output_path = PROJECT_ROOT / "sandeep" / "ned" / "downloads" / "forecast_vs_actual.csv"
model_metrics_results_path = PROJECT_ROOT / "sandeep" / "ned" / "downloads"  / "model_run_metrics.csv"
forecast_output_path_rolling = PROJECT_ROOT / "sandeep" / "ned" / "downloads"  / "forecast_vs_actual-rollingwindow.csv"
model_metrics_results_path_rolling = PROJECT_ROOT / "sandeep" / "ned" / "downloads"  / "model_run_metrics-rollingwindow.csv"

# === Load model ===
model = joblib.load(model_file_path)
logger.info("✅ Prophet model loaded from disk.")

# Step 1: Read data
csv_file_path = PROJECT_ROOT / "sandeep" / "ned" / "downloads"  / "warp-csv-dataset.csv"
with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

df_pd_orig['datetime'] = df_pd_orig['target_datetime']
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
df = df_pd_orig.sort_values(by='datetime')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
df['ds'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
df['y'] = df['Price']

regressors = [
    'Load','shortwave_radiation','temperature_2m','direct_normal_irradiance','diffuse_radiation','Flow_NO','yearday_cos','Flow_GB','month',
    'is_dst','yearday_sin','is_non_working_day','hour_cos','is_weekend','cloud_cover','weekday_sin','hour_sin','weekday_cos'
]

# wind_speed_10m removed.

logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

# === 1. Normal Forecast (single window) ===
forecast_horizon_days = 6
forecast_start = df['ds'].max() - pd.Timedelta(days=forecast_horizon_days-1)
forecast_end = df['ds'].max()

future = df[(df['ds'] >= forecast_start) & (df['ds'] <= forecast_end)].copy()
future_pred = future[['ds'] + regressors].copy()
if future_pred[regressors].isnull().any().any():
    logger.warning("Missing values found in regressors for normal forecast, applying ffill/bfill.")
    future_pred[regressors] = future_pred[regressors].ffill().bfill()

forecast = model.predict(future_pred)
yhat = forecast[['ds', 'yhat']]
actual = future[['ds', 'y']]
merged = actual.merge(yhat, on='ds', how='left')

mae = mean_absolute_error(merged['y'], merged['yhat'])
mse = mean_squared_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mse)
r2 = r2_score(merged['y'], merged['yhat'])

logger.info(f"Normal Forecast metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# Save normal forecast results
df_forecast = merged.rename(columns={'y': 'actual', 'yhat': 'predicted'})
df_forecast.to_csv(forecast_output_path, index=False)
logger.info(f"📁 Normal Forecast vs Actual saved to: {forecast_output_path}")

df_model_metrics = pd.DataFrame([{
    "run": 1,
    "start": forecast_start,
    "end": forecast_end,
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}])
df_model_metrics.to_csv(model_metrics_results_path, index=False)
logger.info(f"Normal forecast metrics saved to: {model_metrics_results_path}")

# === 2. Rolling Window Forecast ===
start_date = pd.to_datetime("2025-05-24") + timedelta(hours=36)
num_rolling_runs = 6
horizon = 6  # days ahead

forecast_rows = []
model_run_metrics = []

for i in range(num_rolling_runs):
    predict_start = start_date + pd.Timedelta(days=i)
    predict_end = predict_start + pd.Timedelta(days=horizon - 1)
    logger.info(f"Rolling window {i+1}: {predict_start} to {predict_end}")

    future = df[(df['ds'] >= predict_start) & (df['ds'] <= predict_end)].copy()
    missing_regs = [r for r in regressors if r not in future.columns]
    if missing_regs:
        logger.error(f"Missing regressors in prediction window: {missing_regs}")
        raise ValueError(f"Missing regressors in prediction window: {missing_regs}")

    future_pred = future[['ds'] + regressors].copy()
    if future_pred[regressors].isnull().any().any():
        logger.warning(f"Missing values found in regressors for window {i+1}, applying ffill/bfill.")
        future_pred[regressors] = future_pred[regressors].ffill().bfill()

    forecast = model.predict(future_pred)
    yhat = forecast[['ds', 'yhat']]
    actual = future[['ds', 'y']]
    merged = actual.merge(yhat, on='ds', how='left')

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    mse = mean_squared_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(merged['y'], merged['yhat'])

    logger.info(f"Window {i+1} metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    for _, row in merged.iterrows():
        forecast_rows.append({
            "ds": row['ds'],
            "actual": row['y'],
            "predicted": row['yhat'],
            "window_start": predict_start,
            "window_end": predict_end,
            "rmse": rmse
        })

    model_run_metrics.append({
        "run": i + 1,
        "start": predict_start,
        "end": predict_end,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    })

# Save rolling window forecast results
try:
    df_forecast_rolling = pd.DataFrame(forecast_rows)
    df_forecast_rolling.to_csv(forecast_output_path_rolling, index=False)
    logger.info(f"📁 Rolling Window Forecast vs Actual saved to: {forecast_output_path_rolling}")
except Exception as e:
    logger.error(f"Failed to save rolling window forecast vs actual: {e}")
    raise

try:
    df_model_metrics_rolling = pd.DataFrame(model_run_metrics)
    df_model_metrics_rolling.to_csv(model_metrics_results_path_rolling, index=False)
    logger.info(f"Rolling window model run metrics saved to: {model_metrics_results_path_rolling}")
except Exception as e:
    logger.error(f"Failed to save rolling window model run metrics: {e}")
    raise
