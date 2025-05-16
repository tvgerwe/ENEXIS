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
csv_file_path = os.path.join(CSV_DATA_DIR, f"prophet_validation_actual_vs_predicted.csv")

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Ensure 'ds' is datetime
df_pd_orig['ds'] = pd.to_datetime(df_pd_orig['ds'])

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(df_pd_orig['ds'], df_pd_orig['y'], label='Actual', color='black')
plt.plot(df_pd_orig['ds'], df_pd_orig['yhat'], label='Predicted', color='dodgerblue', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
