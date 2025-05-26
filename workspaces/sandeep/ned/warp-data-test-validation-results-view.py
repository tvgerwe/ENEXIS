import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import logging

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

CSV_DATA_DIR = config['ned']['ned_model_download_dir']

# --- Plot for forecast_vs_actual.csv ---
csv_file_path = os.path.join(CSV_DATA_DIR, "forecast_vs_actual.csv")
df_normal = pd.read_csv(csv_file_path)
df_normal['ds'] = pd.to_datetime(df_normal['ds'])

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
axs[0].plot(df_normal['ds'], df_normal['actual'], label='Actual', color='green')
axs[0].plot(df_normal['ds'], df_normal['predicted'], label='Predicted', color='blue', linestyle='--')
axs[0].set_ylabel('Value')
axs[0].set_title('Normal Forecast: Actual vs Predicted')
axs[0].legend()
axs[0].grid(True)
if 'rmse' in df_normal.columns:
    axs[1].plot(df_normal['ds'], df_normal['rmse'], label='RMSE', color='red')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('RMSE')
    axs[1].set_title('Normal Forecast: RMSE over Time')
    axs[1].legend()
    axs[1].grid(True)
plt.tight_layout()
plt.show()

# --- Plot for forecast_vs_actual-rollingwindow.csv (averaged by ds) ---
csv_file_path_rolling = os.path.join(CSV_DATA_DIR, "forecast_vs_actual-rollingwindow.csv")
df_rolling = pd.read_csv(csv_file_path_rolling)
df_rolling['ds'] = pd.to_datetime(df_rolling['ds'])

# Group by 'ds' and calculate mean for actual, predicted, and rmse
df_rolling_avg = df_rolling.groupby('ds')[['actual', 'predicted', 'rmse']].mean().reset_index()

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
axs[0].plot(df_rolling_avg['ds'], df_rolling_avg['actual'], label='Actual (avg)', color='green')
axs[0].plot(df_rolling_avg['ds'], df_rolling_avg['predicted'], label='Predicted (avg)', color='blue', linestyle='--')
axs[0].set_ylabel('Value')
axs[0].set_title('Rolling Window Forecast (Averaged): Actual vs Predicted')
axs[0].legend()
axs[0].grid(True)
if 'rmse' in df_rolling_avg.columns:
    axs[1].plot(df_rolling_avg['ds'], df_rolling_avg['rmse'], label='RMSE (avg)', color='red')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('RMSE')
    axs[1].set_title('Rolling Window Forecast (Averaged): RMSE over Time')
    axs[1].legend()
    axs[1].grid(True)
plt.tight_layout()
plt.show()
