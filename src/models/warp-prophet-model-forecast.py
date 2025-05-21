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
import sqlite3

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
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

model_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "prophet_hyper_tuned_model.pkl"
forecast_output_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "forecast_vs_actual.csv"
model_metrics_results_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "model_run_metrics.csv"

# === Load model (Optional, not reused in rolling window) ===
model = joblib.load(model_file_path)
logger.info("✅ Prophet model loaded from disk.")


# Step 1: Read JSON data from a file
csv_file_path = PROJECT_ROOT / "src" / "data" / "warp-csv-dataset.csv"

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Step 1: Convert 'validto' column to datetime
df_pd_orig['datetime'] = df_pd_orig['target_datetime']
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
# Step 2: Sort the DataFrame by 'validto' to avoid data leakage
df = df_pd_orig.sort_values(by='datetime')
# Step 3: Initial datetime formatting
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)  # Ensure no timezone


# === User Input: Set forecast limit date ===
rolling_cutoff_date = pd.to_datetime("2025-05-15")  # Adjust as needed

# === Preprocess Data ===
df['ds'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)  # Ensure tz-naive datetime
df['y'] = df['Price']

# Define regressors you want to use
# regressors = ['Solar_Vol', 'Total_Flow', 'temperature_2m']
regressors = ['month','shortwave_radiation','apparent_temperature','temperature_2m','direct_normal_irradiance','diffuse_radiation','yearday_sin',
              'Flow_BE','hour_sin','is_non_working_day','is_weekend','is_holiday','weekday_cos','wind_speed_10m','hour_cos','weekday_sin',
              'cloud_cover','Flow_GB','Nuclear_Vol','yearday_cos','Flow_NO','Load']


logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

start_date = pd.to_datetime("2025-04-17") + timedelta(hours=36)
num_rolling_runs = 6
horizon = 6  # days ahead
step = 1     # step forward by 1 day for each rolling prediction

results = []
forecast_rows = []
model_run_metrics = []

for i in range(num_rolling_runs):
    predict_start = start_date + pd.Timedelta(days=i)
    predict_end = predict_start + pd.Timedelta(days=horizon - 1)
    logger.info(f"Rolling window {i+1}: {predict_start} to {predict_end}")
    # Select data window for prediction (this includes regressors)
    future = df[(df['ds'] >= predict_start) & (df['ds'] <= predict_end)].copy()

    # Check if all regressors are present
    missing_regs = [r for r in regressors if r not in future.columns]
    if missing_regs:
        logger.error(f"Missing regressors in prediction window: {missing_regs}")
        raise ValueError(f"Missing regressors in prediction window: {missing_regs}")

    # Prepare future dataframe for prediction: must include 'ds' and all regressors
    future_pred = future[['ds'] + regressors]

    # Make sure there are no missing values in regressors (fill or drop)
    if future_pred[regressors].isnull().any().any():
        logger.warning(f"Missing values found in regressors for window {i+1}, applying ffill/bfill.")
        future_pred[regressors] = future_pred[regressors].ffill().bfill()

    # Predict using the model
    forecast = model.predict(future_pred)

    # Extract predicted values and merge with actuals
    yhat = forecast[['ds', 'yhat']]
    actual = future[['ds', 'y']]

    merged = actual.merge(yhat, on='ds', how='left')

    # Evaluate error only where predictions are present
    mae = mean_absolute_error(merged['y'], merged['yhat'])

    # Calculate performance metrics
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    mse = mean_squared_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(merged['y'], merged['yhat'])

    logger.info(f"Window {i+1} metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    # Append actual and predicted rows to list with window info
    for _, row in merged.iterrows():
        forecast_rows.append({
            "ds": row['ds'],
            "actual": row['y'],
            "predicted": row['yhat'],
            "window_start": predict_start,
            "window_end": predict_end
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

# Save actual vs predicted to CSV
try:
    df_forecast = pd.DataFrame(forecast_rows)
    df_forecast.to_csv(forecast_output_path, index=False)
    logger.info(f"📁 Forecast vs Actual saved to: {forecast_output_path}")
    logger.info("\n" + str(df_forecast.head()))
except Exception as e:
    logger.error(f"Failed to save forecast vs actual: {e}")
    raise

# === Save Results ===
try:
    df_model_metrics = pd.DataFrame(model_run_metrics)
    df_model_metrics.to_csv(model_metrics_results_path, index=False)
    logger.info(f"Model run metrics saved to: {model_metrics_results_path}")
except Exception as e:
    logger.error(f"Failed to save model run metrics: {e}")
    raise
