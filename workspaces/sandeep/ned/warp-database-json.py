# env - enexis-may-03-env-run

import pandas as pd
import os
from datetime import datetime
import time

import sqlite3

from pathlib import Path
import logging
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('To fetch data from database and save as csv')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

# === CONFIG ===

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_RUN_RESULTS_DIR = config['ned']['ned_model_download_dir']

# Connect to the SQLite database
db_path = '/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/WARP.db'
conn = sqlite3.connect(db_path)
# Connect to the SQLite database using the existing db_path
conn = sqlite3.connect(db_path)
# Step 2: Read data from table
df_pd_orig = pd.read_sql_query("SELECT * FROM master_warp ORDER BY target_datetime DESC", conn)

# df_pd_orig = pd.read_sql_query("SELECT * FROM raw_entsoe_obs ORDER BY Timestamp DESC", conn)
#df_pd_orig["datetime"] = df_pd_orig["Timestamp"]
# Step 3: Close the connection
conn.close()

logger = logging.getLogger('Database fetch complete')

db_results_file_path = f'{MODEL_RUN_RESULTS_DIR}warp-csv-dataset.csv'

# Step 10: Check if file exists, then append or create
if os.path.exists(db_results_file_path):
    # Append to existing file
    existing_results = pd.read_csv(db_results_file_path)
    updated_results = pd.concat([existing_results, df_pd_orig], ignore_index=True)
    updated_results.to_csv(db_results_file_path, index=False)
else:
    # Create new file
    df_pd_orig.to_csv(db_results_file_path, index=False)

logger = logging.getLogger('csv save complete at : ' + db_results_file_path)
