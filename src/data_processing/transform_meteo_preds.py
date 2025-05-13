#!/usr/bin/env python3

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import timedelta

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - transform_meteo_preds_history - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transform_meteo_preds_history")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
SOURCE_TABLE = "raw_meteo_preds_history"
OUTPUT_TABLE = "transform_weather_preds_history"

def load_preds(conn):
    logger.info(f"📥 Laden van tabel: {SOURCE_TABLE}")
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE}", conn)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df

def transform_to_long_format(df):
    logger.info("🔄 Transformeren naar long-format met fetch_date (datum) + forecast_for")

    long_rows = []

    for col in df.columns:
        if col.startswith("temperature_2m_previous_day"):
            day_offset = int(col.replace("temperature_2m_previous_day", ""))
            temp_df = df[["date", col]].copy()
            temp_df = temp_df.rename(columns={col: "temperature_2m"})
            temp_df["forecast_for"] = temp_df["date"]
            temp_df["fetch_date"] = (temp_df["forecast_for"] - timedelta(days=day_offset)).dt.floor("D")
            long_rows.append(temp_df[["fetch_date", "forecast_for", "temperature_2m"]])

    combined = pd.concat(long_rows, ignore_index=True)
    combined = combined.sort_values(["forecast_for", "fetch_date"]).reset_index(drop=True)

    logger.info(f"✅ {len(combined)} rijen in long-format gegenereerd")
    return combined

def main():
    logger.info(f"📦 Verbinden met database: {DB_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"❌ Database niet gevonden: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        df_preds = load_preds(conn)
        df_long = transform_to_long_format(df_preds)

        logger.info(f"💾 Wegschrijven naar tabel: {OUTPUT_TABLE}")
        df_long.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

        logger.info("🎉 transform_meteo_preds_history succesvol opgeslagen.")
    except Exception as e:
        logger.error(f"❌ Fout bij transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    main()