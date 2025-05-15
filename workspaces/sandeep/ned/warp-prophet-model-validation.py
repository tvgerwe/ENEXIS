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

# === Load model (Optional, not reused in rolling window) ===
model = joblib.load(model_file_path)
logger.info("✅ Prophet model loaded from disk.")

# === Load data from SQLite ===
db_path = '/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/WARP.db'
conn = sqlite3.connect(db_path)
df_raw = pd.read_sql_query("SELECT * FROM master_warp ORDER BY datetime DESC", conn)
conn.close()


# Prepare data
df_raw['datetime'] = pd.to_datetime(df_raw['datetime']).dt.tz_localize(None)

val_start = pd.to_datetime("2025-04-15")
val_end = pd.to_datetime("2025-05-14")

# Filter raw data for validation period
df_filtered = df_raw[(df_raw['datetime'] >= val_start) & (df_raw['datetime'] <= val_end)].copy()

# Filter positive Price and prepare target
# df = df_filtered[df_filtered['Price'] > 0].copy()
df = df_filtered.copy()
df['ds'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
df['y'] = df['Price']
df = df[['ds', 'y']].sort_values('ds')

logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

# Prepare regressors DataFrame (make sure columns exist and are cleaned)
regressors = ['Total_Flow', 'Solar_Vol', 'Wind_Vol', 'WindOffshore_Vol', 'Nuclear_Vol', 'temperature_2m']
X = df_filtered[['datetime'] + regressors].copy()
X['ds'] = pd.to_datetime(X['datetime']).dt.tz_localize(None)
X.drop(columns=['datetime'], inplace=True)
X.dropna(subset=regressors, inplace=True)

# Manually create future DataFrame for validation period only
future_val = pd.DataFrame({'ds': pd.date_range(start=val_start, end=val_end, freq='D')})

# Merge regressors onto future_val on 'ds'
future_val = future_val.merge(X, on='ds', how='left')

# Fill missing regressor values (forward then backward fill)
future_val[regressors] = future_val[regressors].ffill().bfill()

print(f"Future validation dataframe length: {len(future_val)}")
print(f"Number of missing regressor values after fill: {future_val[regressors].isna().sum().sum()}")

if future_val.empty or future_val[regressors].isna().all().all():
    raise ValueError("Future dataframe for validation period is empty or missing regressors. Check your data.")

# Predict with model on manual future_val dataframe
forecast = model.predict(future_val)

# Filter actual for validation period and set index on 'ds'
actual = df.set_index('ds')

print(f"Actual data length in validation period: {len(actual)}")

if actual.empty:
    raise ValueError("Actual data for validation period is empty. Check your dataset and val_start/val_end.")

# Join actual and predicted data on 'ds', drop NA rows
merged = actual.join(forecast.set_index('ds')[['yhat']], how='inner').dropna().reset_index()

print(f"Merged dataframe length: {len(merged)}")

if merged.empty:
    raise ValueError("No overlapping rows between actual and predicted data in the validation period.")

# Calculate evaluation metrics
mae = mean_absolute_error(merged['y'], merged['yhat'])
mse = mean_squared_error(merged['y'], merged['yhat'])
rmse = mse ** 0.5
r2 = r2_score(merged['y'], merged['yhat'])

print(f"Validation metrics from {val_start.date()} to {val_end.date()}:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")




# === Save Actual vs Predicted to CSV ===
output_file = f'{MODEL_RUN_RESULTS_DIR}prophet_validation_actual_vs_predicted.csv'
merged[['ds', 'y', 'yhat']].to_csv(output_file, index=False)
logger.info(f"✅ Actual vs Predicted values saved to {output_file}")


# === Save Results ===
validation_output_path = f'{MODEL_RUN_RESULTS_DIR}validation_results.csv'
merged.reset_index().to_csv(validation_output_path, index=False)
logger.info(f"💾 Forecast vs Actual saved to {validation_output_path}")
