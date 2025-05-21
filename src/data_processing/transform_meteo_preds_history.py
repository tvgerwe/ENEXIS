#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path
import re
from functools import reduce
from datetime import timedelta

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - transform_meteo_preds_history - %(levelname)s - %(message)s"
)
logger = logging.getLogger("transform_meteo_preds_history")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
RAW_TABLE = "raw_weather_preds"
TRANSFORM_TABLE = "process_weather_preds"

def transform():
    conn = sqlite3.connect(DB_PATH)
    try:
        # First, check if raw table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (RAW_TABLE,))
        if not cursor.fetchone():
            logger.error(f"❌ Tabel {RAW_TABLE} bestaat niet in de database")
            return

        # Count rows in raw table
        cursor.execute(f"SELECT COUNT(*) FROM {RAW_TABLE}")
        raw_count = cursor.fetchone()[0]
        if raw_count == 0:
            logger.error(f"❌ Tabel {RAW_TABLE} bevat geen data")
            return
        
        logger.info(f"ℹ️ Tabel {RAW_TABLE} bevat {raw_count} rijen")
        
        # Load raw data
        df = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", conn)
        logger.info(f"✅ {RAW_TABLE} geladen ({len(df)} rijen)")

        # Check and report NULL values in _previous_day columns
        previous_day_cols = [col for col in df.columns if '_previous_day' in col]
        null_counts = df[previous_day_cols].isnull().sum()
        
        # Log warning if many NULL values are found
        high_null_cols = null_counts[null_counts > len(df) * 0.5].index.tolist()
        if high_null_cols:
            logger.warning(f"⚠️ Veel NULL-waarden gevonden in {len(high_null_cols)} kolommen")
            logger.debug(f"Kolommen met >50% NULL: {high_null_cols[:5]}...")

        df["target_datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.drop(columns=["date"])

        # Extract variable base names and their "previous day" variants
        pattern = re.compile(r"^(?P<varname>.+)_previous_day(?P<day>\d+)$")
        variable_map = {}
        base_variables = set()

        # First, identify all variables with _previous_day pattern
        for col in df.columns:
            match = pattern.match(col)
            if match:
                varname = match.group("varname")
                day = int(match.group("day"))
                base_variables.add(varname)
                variable_map.setdefault(varname, []).append((day, col))

        # Next, add current day variables (those without _previous_day suffix)
        for col in df.columns:
            if col != "target_datetime" and "_previous_day" not in col:
                # Check if this column is a base variable that may have previous_day variants
                if col in base_variables:
                    # Add as day 0 (current day's value)
                    variable_map.setdefault(col, []).append((0, col))
                    logger.info(f"✅ Toegevoegd huidige waarde: {col} (dag 0)")

        logger.info(f"📊 Geïdentificeerde variabelen: {list(variable_map.keys())}")

        result_frames = []

        for var, entries in variable_map.items():
            rows = []
            for horizon, col in sorted(entries):
                temp = df[["target_datetime", col]].copy()
                # Calculate run_date (when the prediction was made)
                temp["run_date"] = (temp["target_datetime"] - pd.Timedelta(days=horizon)).dt.normalize()
                temp.rename(columns={col: var}, inplace=True)
                
                # Only include rows where the variable has a value
                temp = temp.dropna(subset=[var])
                if not temp.empty:
                    rows.append(temp)
                    logger.info(f"✅ Verwerkt {var} voor dag {horizon}: {len(temp)} rijen")
                else:
                    logger.warning(f"⚠️ Geen data voor {var} dag {horizon}")

            if rows:
                merged = pd.concat(rows)
                result_frames.append(merged)
            else:
                logger.warning(f"⚠️ Geen rijen voor variabele {var}")

        if not result_frames:
            logger.error("❌ Geen data om te transformeren")
            return

        df_final = reduce(
            lambda left, right: pd.merge(left, right, on=["run_date", "target_datetime"], how="outer"),
            result_frames
        )

        # ✅ Forceer correct datetime-type
        df_final["run_date"] = pd.to_datetime(df_final["run_date"], utc=True, errors="coerce")
        df_final["target_datetime"] = pd.to_datetime(df_final["target_datetime"], utc=True, errors="coerce")

        if df_final["run_date"].isnull().any():
            logger.warning("⚠️ Ongeldige run_date waarden aangetroffen (NaT)")

        if df_final["target_datetime"].isnull().any():
            logger.warning("⚠️ Ongeldige target_datetime waarden aangetroffen (NaT)")

        # Don't drop rows where all weather variables are NULL, since we want to include today's values
        # Just make sure we have at least one of run_date or target_datetime
        df_final = df_final.dropna(subset=["run_date", "target_datetime"], how="all")
        df_final = df_final.sort_values(["target_datetime", "run_date"])

        logger.info(f"📊 Transformatie klaar: {df_final.shape[0]} rijen, {df_final.shape[1]} kolommen")
        
        # Check if transformation produced data
        if df_final.empty:
            logger.error("❌ Transformatie resulteerde in een lege dataset")
            return
            
        df_final.to_sql(TRANSFORM_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ Weggeschreven naar {TRANSFORM_TABLE}")
        
        # Confirm data was saved
        cursor.execute(f"SELECT COUNT(*) FROM {TRANSFORM_TABLE}")
        transform_count = cursor.fetchone()[0]
        logger.info(f"✅ {TRANSFORM_TABLE} bevat nu {transform_count} rijen")
        
    except Exception as e:
        logger.error(f"❌ Fout tijdens transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    transform()