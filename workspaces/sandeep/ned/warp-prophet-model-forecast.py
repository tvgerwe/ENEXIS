import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from pathlib import Path
import logging
import json
import sqlite3

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prophet_rolling_validation')

# === Config Setup ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_RUN_RESULTS_DIR = config['ned']['ned_model_download_dir']
model_file_path = f'{MODEL_RUN_RESULTS_DIR}prophet_model.pkl'
rolling_window_file_path = f'{MODEL_RUN_RESULTS_DIR}rolling_validation_results.csv'
forecast_output_path = f"{MODEL_RUN_RESULTS_DIR}/forecast_vs_actual.csv"

# === Load model (Optional, not reused in rolling window) ===
model = joblib.load(model_file_path)
logger.info("✅ Prophet model loaded from disk.")

CSV_DATA_DIR = config['ned']['ned_model_download_dir']

# Step 1: Read JSON data from a file
csv_file_path = os.path.join(CSV_DATA_DIR, f"warp-csv-dataset.csv")

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Step 1: Convert 'validto' column to datetime
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
regressors = ['Solar_Vol', 'Total_Flow', 'temperature_2m']

logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

# Rolling forecast parameters
start_date = pd.to_datetime("2025-04-15")
num_rolling_runs = 6
horizon = 6  # days ahead

results = []
forecast_rows = []

for i in range(num_rolling_runs):
    predict_start = start_date + pd.Timedelta(days=i)
    predict_end = predict_start + pd.Timedelta(days=horizon - 1)
    # Select data window for prediction (this includes regressors)
    future = df[(df['ds'] >= predict_start) & (df['ds'] <= predict_end)].copy()

    # Check if all regressors are present
    missing_regs = [r for r in regressors if r not in future.columns]
    if missing_regs:
        raise ValueError(f"Missing regressors in prediction window: {missing_regs}")

    # Prepare future dataframe for prediction: must include 'ds' and all regressors
    future_pred = future[['ds'] + regressors]

    # Make sure there are no missing values in regressors (fill or drop)
    if future_pred[regressors].isnull().any().any():
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

    # Append actual and predicted rows to list with window info
    for _, row in merged.iterrows():
        forecast_rows.append({
            "ds": row['ds'],
            "actual": row['y'],
            "predicted": row['yhat'],
            "window_start": predict_start,
            "window_end": predict_end
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Save actual vs predicted to CSV
forecast_output_path = "forecast_vs_actual.csv"  # Change path as needed
df_forecast = pd.DataFrame(forecast_rows)
df_forecast.to_csv(forecast_output_path, index=False)
logger.info(f"📁 Forecast vs Actual saved to: {forecast_output_path}")
logger.info("\n" + str(df_forecast.head()))

# === Save Results ===
df_results = pd.DataFrame(results)
df_results.to_csv(rolling_window_file_path, index=False)

# === Save Actual vs Predicted CSV ===
df_forecast = pd.DataFrame(forecast_rows)

df_forecast.to_csv(forecast_output_path, index=False)
logger.info(f"📁 Forecast vs Actual saved to: {forecast_output_path}")
logger.info("\n" + str(df_forecast.head()))

logger.info(f"📊 Rolling validation results saved to: {rolling_window_file_path}")
logger.info("\n" + str(df_results.head()))
