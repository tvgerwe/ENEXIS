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
        
        # DEBUG: Check available run_dates before filtering
        available_run_dates = df_preds["run_date"].unique()
        logger.info(f"📊 Available run_dates: {[str(d) for d in available_run_dates[:5]]}... (showing first 5)")
        logger.info(f"🔍 Looking for run_date: {run_date}")
        
        # SOLUTION 1: Use date() method to compare only the date part if timestamps don't match exactly
        run_date_only = run_date.date()
        df_preds_with_date = df_preds.copy()
        df_preds_with_date["run_date_only"] = df_preds["run_date"].dt.date
        df_preds = df_preds_with_date[
            (df_preds_with_date["run_date_only"] == run_date_only) &
            (df_preds_with_date["target_datetime"] >= forecast_start) &
            (df_preds_with_date["target_datetime"] <= forecast_end)
        ]
        
        # If still empty, try alternative approach with nearest run_date
        if df_preds.empty:
            logger.warning("⚠️ No exact matches for run_date. Trying to find the closest run_date...")
            
            # Find the closest run_date available (preferably before our target run_date)
            closest_before = df_preds_with_date[df_preds_with_date["run_date"] <= run_date]
            
            if not closest_before.empty:
                max_run_date = closest_before["run_date"].max()
                logger.info(f"📅 Using closest available run_date: {max_run_date}")
                
                df_preds = df_preds_with_date[
                    (df_preds_with_date["run_date"] == max_run_date) &
                    (df_preds_with_date["target_datetime"] >= forecast_start) &
                    (df_preds_with_date["target_datetime"] <= forecast_end)
                ]
        
        logger.info(f"✅ Forecast geladen: {df_preds.shape[0]} rijen voor run_date {run_date.date()}")

        '''# === Voeg 'Price' toe aan df_preds vanuit df_actuals op basis van target_datetime ===
        df_actuals_forecast = pd.read_sql_query(f"""
            SELECT target_datetime, Price 
            FROM {ACTUALS_TABLE}
            WHERE target_datetime BETWEEN ? AND ?
        """, conn, params=(forecast_start.isoformat(), forecast_end.isoformat()))
        df_actuals_forecast["target_datetime"] = pd.to_datetime(df_actuals_forecast["target_datetime"], utc=True)

        # Merge actual prices into df_preds
        df_preds = df_preds.merge(df_actuals_forecast, on="target_datetime", how="left", suffixes=('', '_actual'))

        # Als er al een 'Price' kolom in df_preds zat, vervang die nu met de echte waarde (indien beschikbaar)
        df_preds["Price"] = df_preds["Price_actual"].combine_first(df_preds["Price"])
        df_preds = df_preds.drop(columns=["Price_actual"])

        # Drop the temporary column if we added it'''
        if "run_date_only" in df_preds.columns:
            df_preds = df_preds.drop(columns=["run_date_only"])

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