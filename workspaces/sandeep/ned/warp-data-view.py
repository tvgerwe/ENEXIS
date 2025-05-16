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
csv_file_path = os.path.join(CSV_DATA_DIR, f"warp-csv-dataset.csv")

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Ensure 'datetime' is in datetime format
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])

# Set up subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot each variable in its own subplot
axs[0].plot(df_pd_orig['datetime'], df_pd_orig['Total_Flow'], color='blue')
axs[0].set_title('Total Flow')
axs[0].grid(True)

axs[1].plot(df_pd_orig['datetime'], df_pd_orig['Solar_Vol'], color='orange')
axs[1].set_title('Solar Volume')
axs[1].grid(True)

axs[2].plot(df_pd_orig['datetime'], df_pd_orig['temperature_2m'], color='green')
axs[2].set_title('Temperature 2m')
axs[2].set_xlabel('Datetime')
axs[2].grid(True)

# Improve layout
plt.tight_layout()
plt.show()
