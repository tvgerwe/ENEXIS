#!/usr/bin/env python3

import sqlite3
import pandas as pd
import logging
from pathlib import Path

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - transform_meteo_forecast_now - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transform_meteo_forecast_now")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
RAW_TABLE = "raw_meteo_forecast_now"
OBS_TABLE = "raw_meteo_obs"
OUTPUT_TABLE = "transform_meteo_forecast_now"

def get_connection(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Database niet gevonden: {path}")
    return sqlite3.connect(path)

def transform():
    logger.info(f"📦 Verbinden met database: {DB_PATH}")
    conn = get_connection(DB_PATH)

    try:
        logger.info(f"📥 Laden van voorspellingen uit {RAW_TABLE}")
        df = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", conn)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info(f"📥 Ophalen van laatste observatie uit {OBS_TABLE}")
        obs_df = pd.read_sql_query(f"SELECT MAX(date) as max_obs FROM {OBS_TABLE}", conn)
        max_obs_date = pd.to_datetime(obs_df["max_obs"][0])
        logger.info(f"⏩ Filtering vanaf: {max_obs_date}")

        # Filter voorspellingen die ná de laatste observatie vallen
        df = df[df["date"] > max_obs_date]
        df = df.sort_values("date")

        logger.info(f"💾 Wegschrijven naar tabel: {OUTPUT_TABLE} ({len(df)} rijen)")
        df.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

        logger.info("🎉 transform_meteo_forecast_now succesvol opgeslagen.")
    except Exception as e:
        logger.error(f"❌ Fout bij transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    transform()