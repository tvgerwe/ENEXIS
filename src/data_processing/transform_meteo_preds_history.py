#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - transform_meteo_preds_history - %(levelname)s - %(message)s"
)
logger = logging.getLogger("transform_meteo_preds_history")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
RAW_TABLE = "raw_meteo_preds_history"
TRANSFORM_TABLE = "transform_weather_preds_history"

def get_connection(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Database niet gevonden: {path}")
    return sqlite3.connect(path)

def transform():
    logger.info(f"📦 Gebruik van database: {DB_PATH}")
    conn = get_connection(DB_PATH)

    try:
        logger.info(f"📥 Ophalen van data uit {RAW_TABLE}")
        df = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", conn)

        logger.info("🔧 Start transformatie")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.drop_duplicates(subset="date", keep="last")
        df = df.sort_values("date")

        logger.info(f"💾 Overschrijven van {TRANSFORM_TABLE}")
        df.to_sql(TRANSFORM_TABLE, conn, if_exists="replace", index=False)

        logger.info(f"✅ {TRANSFORM_TABLE} bevat {len(df)} rijen")
    except Exception as e:
        logger.error(f"❌ Fout tijdens transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    transform()