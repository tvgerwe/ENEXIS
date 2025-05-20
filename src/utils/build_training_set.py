#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# === Logging configuratie ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - build_training_set - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_training_set")

# === Config ===
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "WARP.db"
OUTPUT_TABLE = "training_set"
ACTUALS_TABLE = "master_warp"
PREDICTIONS_TABLE = "master_predictions"
HORIZON = 168

# === Periode instellen ===
train_start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
train_end = pd.Timestamp("2025-03-14 23:00:00", tz="UTC")
run_date = train_end + pd.Timedelta(hours=1)
forecast_start = run_date
forecast_end = forecast_start + pd.Timedelta(hours=HORIZON - 1)

def build_training_set():
    logger.info("🚀 Start build van trainingset")
    logger.info(f"🧠 Actuals van {train_start} t/m {train_end}")
    logger.info(f"📅 Forecast van run_date {run_date}, target range: {forecast_start} → {forecast_end}")

    conn = sqlite3.connect(DB_PATH)

    try:
        # === Load actuals ===
        df_actuals = pd.read_sql_query(f"SELECT * FROM {ACTUALS_TABLE}", conn)
        df_actuals["target_datetime"] = pd.to_datetime(df_actuals["target_datetime"], utc=True)
        df_actuals = df_actuals[
            (df_actuals["target_datetime"] >= train_start) &
            (df_actuals["target_datetime"] <= train_end)
        ]
        logger.info(f"✅ master_warp geladen: {df_actuals.shape[0]} rijen")

        # === Load forecasts ===
        df_preds = pd.read_sql_query(f"SELECT * FROM {PREDICTIONS_TABLE}", conn)
        df_preds["target_datetime"] = pd.to_datetime(df_preds["target_datetime"], utc=True)
        df_preds["run_date"] = pd.to_datetime(df_preds["run_date"], utc=True)

        df_preds = df_preds[
            (df_preds["run_date"] == run_date) &
            (df_preds["target_datetime"] >= forecast_start) &
            (df_preds["target_datetime"] <= forecast_end)
        ]
        logger.info(f"✅ Forecast geladen: {df_preds.shape[0]} rijen voor run_date {run_date.date()}")

        # === Combineer via concat (geen merge, geen suffix!)
        df = pd.concat([df_actuals, df_preds], ignore_index=True)
        df = df.sort_values("target_datetime").drop_duplicates("target_datetime")

        logger.info(f"📦 Eindtabel bevat: {df.shape[0]} rijen, {df.shape[1]} kolommen")
        logger.info(f"🧾 Kolommen: {df.columns.tolist()}")

        df.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ Opgeslagen als {OUTPUT_TABLE} in {DB_PATH.name}")

    except Exception as e:
        logger.error(f"❌ Fout tijdens build: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    build_training_set()
