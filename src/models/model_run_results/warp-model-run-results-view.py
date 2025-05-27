import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error
import numpy as np

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prophet_rolling_validation')

# === Config Setup ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

forecast_output_path = PROJECT_ROOT / "models" / "model_run_results" / "forecast_vs_actual.csv"
forecast_output_path_rolling = PROJECT_ROOT / "models" / "model_run_results" / "forecast_vs_actual_rolling.csv"
model_metrics_results_path = PROJECT_ROOT / "models" / "model_run_results" / "model_run_metrics.csv"
model_metrics_results_path_rolling = PROJECT_ROOT / "models" / "model_run_results" / "model_run_metrics-rolling.csv"


# --- Plot for forecast_vs_actual.csv ---
df_normal = pd.read_csv(forecast_output_path)
df_normal['ds'] = pd.to_datetime(df_normal['ds'])

# Calculate overall RMSE for normal forecast
if 'actual' in df_normal.columns and 'predicted' in df_normal.columns:
    overall_rmse = np.sqrt(mean_squared_error(df_normal['actual'], df_normal['predicted']))
else:
    overall_rmse = None

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
axs[0].plot(df_normal['ds'], df_normal['actual'], label='Actual', color='green')
axs[0].plot(df_normal['ds'], df_normal['predicted'], label='Predicted', color='blue', linestyle='--')
axs[0].set_ylabel('Value')
title_str = 'Normal Forecast: Actual vs Predicted'
if overall_rmse is not None:
    title_str += f' (RMSE={overall_rmse:.3f})'
axs[0].set_title(title_str)
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
csv_file_path_rolling = os.path.join(forecast_output_path_rolling)
df_rolling = pd.read_csv(csv_file_path_rolling)
df_rolling['ds'] = pd.to_datetime(df_rolling['ds'])

# Group by 'ds' and calculate mean for actual, predicted, and rmse
df_rolling_avg = df_rolling.groupby('ds')[['actual', 'predicted', 'rmse']].mean().reset_index()

# Calculate overall RMSE for rolling forecast
if 'actual' in df_rolling_avg.columns and 'predicted' in df_rolling_avg.columns:
    overall_rmse_rolling = np.sqrt(mean_squared_error(df_rolling_avg['actual'], df_rolling_avg['predicted']))
else:
    overall_rmse_rolling = None

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
axs[0].plot(df_rolling_avg['ds'], df_rolling_avg['actual'], label='Actual (avg)', color='green')
axs[0].plot(df_rolling_avg['ds'], df_rolling_avg['predicted'], label='Predicted (avg)', color='blue', linestyle='--')
axs[0].set_ylabel('Value')
title_str = 'Rolling Window Forecast (Averaged): Actual vs Predicted'
if overall_rmse_rolling is not None:
    title_str += f' (RMSE={overall_rmse_rolling:.3f})'
axs[0].set_title(title_str)
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
