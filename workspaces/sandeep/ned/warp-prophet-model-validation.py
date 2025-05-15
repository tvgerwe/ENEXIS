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

df_raw['datetime'] = pd.to_datetime(df_raw['datetime']).dt.tz_localize(None)

df = df_raw[df_raw['Price'] > 0].copy()
df['ds'] = df['datetime']
df['y'] = df['Price']
df = df[['ds', 'y']].sort_values(by='ds')

logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

# === Define Validation Date Range ===
val_start = "2025-04-15"
val_end   = "2025-05-14"

# Create Future DataFrame and Predict
future = model.make_future_dataframe(periods=(pd.to_datetime(val_end) - df['ds'].min()).days + 1, freq='D')
forecast = model.predict(future)

# Filter actual and predicted for validation window
actual = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)].copy().set_index('ds')
pred = forecast[(forecast['ds'] >= val_start) & (forecast['ds'] <= val_end)].copy().set_index('ds')

# === Join and Evaluate ===
merged = actual.join(pred[['yhat']], how='inner').dropna()
merged = merged.reset_index()  # Bring 'ds' back as a column

if merged.empty:
    logger.warning("❌ No overlapping rows between actual and predicted data in the specified date range.")
else:
    # Evaluate
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    mse = mean_squared_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(merged['y'], merged['yhat'])

    logger.info(f"📊 Validation from {val_start} to {val_end}:")
    logger.info(f"MAE  : {mae:.2f}")
    logger.info(f"MSE  : {mse:.2f}")
    logger.info(f"RMSE : {rmse:.2f}")
    logger.info(f"R²   : {r2:.4f}")

    # === Save Actual vs Predicted to CSV ===
    output_file = f'{MODEL_RUN_RESULTS_DIR}prophet_validation_actual_vs_predicted.csv'
    merged[['ds', 'y', 'yhat']].to_csv(output_file, index=False)
    logger.info(f"✅ Actual vs Predicted values saved to {output_file}")


    # === Save Results ===
    validation_output_path = f'{MODEL_RUN_RESULTS_DIR}validation_results.csv'
    merged.reset_index().to_csv(validation_output_path, index=False)
    logger.info(f"💾 Forecast vs Actual saved to {validation_output_path}")
