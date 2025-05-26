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

# Step 1: Read JSON data from a file
csv_file_path = os.path.join(CSV_DATA_DIR, f"validation_results.csv")

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Ensure 'ds' is datetime
df_pd_orig['ds'] = pd.to_datetime(df_pd_orig['ds'])

# Plot actual vs predicted and RMSE values in a single window with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Actual vs Predicted
axs[0].plot(df_pd_orig['ds'], df_pd_orig['actual'], label='Actual', color='green')
axs[0].plot(df_pd_orig['ds'], df_pd_orig['predicted'], label='Predicted', color='blue', linestyle='--')
axs[0].set_ylabel('Value')
axs[0].set_title('Actual vs Predicted')
axs[0].legend()
axs[0].grid(True)

# RMSE over Time (if present)
if 'rmse' in df_pd_orig.columns:
    axs[1].plot(df_pd_orig['ds'], df_pd_orig['rmse'], label='RMSE', color='red')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('RMSE')
    axs[1].set_title('RMSE over Time')
    axs[1].legend()
    axs[1].grid(True)

plt.tight_layout()
plt.show()
