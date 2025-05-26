# env - enexis-may-03-env-run

import pandas as pd
import os
from datetime import datetime
import time
import sqlite3
from pathlib import Path
import logging
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(Path(__file__).parent / "warp-database-json.log"),
    filemode='a'
)
logger = logging.getLogger('warp-database-json')

CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

# === CONFIG ===
if not CONFIG_PATH.exists():
    logger.error(f"❌ Config not found at : {CONFIG_PATH}")
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_RUN_RESULTS_DIR = config['ned']['ned_model_download_dir']
logger.info(f"MODEL_RUN_RESULTS_DIR: {MODEL_RUN_RESULTS_DIR}")

import sys
from pathlib import Path

# Add project root to sys.path so 'src' is importable
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.utils.build_training_set import build_training_set

build_training_set(
    train_start = "2025-01-01 00:00:00",
    train_end = "2025-03-14 23:00:00",
    run_date="2025-03-23 00:00:00"
)

# Connect to the SQLite database
try:
    db_path = '/Users/sgawde/work/eaisi-code/main-branch-21may/ENEXIS/src/data/WARP.db'
    conn = sqlite3.connect(db_path)
    logger.info(f"Connected to database at {db_path}")
    # Step 2: Read data from table
    df_pd_orig = pd.read_sql_query("SELECT * FROM master_warp ORDER BY target_datetime DESC", conn)
    logger.info("Database fetch complete")
except Exception as e:
    logger.error(f"Database connection or fetch failed: {e}")
    raise
finally:
    conn.close()
    logger.info("Database connection closed")

db_results_file_path = f'{MODEL_RUN_RESULTS_DIR}warp-csv-dataset.csv'

# Step 10: Check if file exists, then append or create
try:
    if os.path.exists(db_results_file_path):
        # Append to existing file
        existing_results = pd.read_csv(db_results_file_path)
        updated_results = pd.concat([existing_results, df_pd_orig], ignore_index=True)
        updated_results.to_csv(db_results_file_path, index=False)
        logger.info(f"Appended and saved to {db_results_file_path}")
    else:
        # Create new file
        df_pd_orig.to_csv(db_results_file_path, index=False)
        logger.info(f"Created new CSV at {db_results_file_path}")
except Exception as e:
    logger.error(f"CSV save failed: {e}")
    raise
