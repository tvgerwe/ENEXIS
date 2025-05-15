import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# === Load data from SQLite ===
db_path = '/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/WARP.db'
conn = sqlite3.connect(db_path)
df_raw = pd.read_sql_query("SELECT * FROM master_warp ORDER BY datetime DESC", conn)
conn.close()

from datetime import datetime

# === User Input: Set forecast limit date ===
rolling_cutoff_date = "2025-05-15"  # <-- Change this date as needed
rolling_cutoff_date = pd.to_datetime(rolling_cutoff_date)

# === Preprocess Data ===
df_raw['datetime'] = pd.to_datetime(df_raw['datetime']).dt.tz_localize(None)
df = df_raw[df_raw['Price'] > 0].copy()
df['ds'] = df['datetime']
df['y'] = df['Price']
df = df[['ds', 'y']].sort_values(by='ds')

logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

# === Rolling Forecast Parameters ===
window_size = 365  # training days
horizon = 30       # forecast horizon
step_size = 30     # rolling step
results = []
forecast_rows = []  # To collect actual vs predicted per date


logger.info("🚀 Starting rolling window validation...")

# Iterate with cutoff logic
for start in range(0, len(df) - window_size - horizon + 1, step_size):
    train_df = df.iloc[start:start + window_size].copy()
    test_df = df.iloc[start + window_size:start + window_size + horizon].copy()

    # Check if test end exceeds cutoff
    test_end_date = test_df['ds'].max()
    if test_end_date > rolling_cutoff_date:
        logger.info(f"🛑 Skipping window: test end {test_end_date.date()} exceeds cutoff {rolling_cutoff_date.date()}")
        break

    # Train and forecast
    model_rolling = Prophet()
    model_rolling.fit(train_df)

    future = model_rolling.make_future_dataframe(periods=horizon, freq='D')
    forecast = model_rolling.predict(future)

    pred = forecast[['ds', 'yhat']].set_index('ds')
    actual = test_df.set_index('ds')

    merged = actual.join(pred, how='inner').dropna()

    if merged.empty:
        logger.warning(f"⚠️ Skipping window {train_df['ds'].min().date()} to {train_df['ds'].max().date()} - no overlap in prediction vs actual.")
        continue

    # Save individual prediction rows
    for row in merged.itertuples():
        forecast_rows.append({
        "date": row.Index,                # the 'ds' date
        "actual": row.y,
        "predicted": row.yhat,
        "train_window_start": train_df['ds'].min(),
        "train_window_end": train_df['ds'].max()
    })


    # Metrics
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    mse = mean_squared_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(merged['y'], merged['yhat'])

    results.append({
        "window_start": train_df['ds'].min(),
        "window_end": train_df['ds'].max(),
        "test_end": test_end_date,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    logger.info(f"✅ Window {train_df['ds'].min().date()} to {train_df['ds'].max().date()} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

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
