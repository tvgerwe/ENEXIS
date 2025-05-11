#!/usr/bin/env python3

import sqlite3
import pandas as pd
import logging
from pathlib import Path

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - transform_ned - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transform_ned')

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
RAW_TABLE = "raw_ned_obs"
TRANSFORM_TABLE = "transform_ned_obs"

def clean_ned_obs(df):
    if 'validto' not in df.columns or 'volume' not in df.columns:
        raise KeyError("⚠️ Vereiste kolommen 'validto' en 'volume' ontbreken in raw_ned_obs.")

    df = df[['volume', 'validto']].copy()
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')
    df['validto'] = pd.to_datetime(df['validto'], utc=True)
    return df.rename(columns=lambda c: f'ned.{c}')

def transform_ned_pipeline():
    logger.info(f"📦 Using DB at: {DB_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"❌ Database niet gevonden op: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        logger.info(f"📥 Ophalen van {RAW_TABLE}")
        df = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", conn)
        if df.empty:
            logger.warning("⚠️ Geen data gevonden in raw_ned_obs. Pipeline stopt.")
            return

        logger.info("🔧 Start transformatie")
        df_transformed = clean_ned_obs(df)

        logger.info(f"🧱 Overschrijven van {TRANSFORM_TABLE}")
        df_transformed.to_sql(TRANSFORM_TABLE, conn, if_exists='replace', index=False)

        logger.info(f"✅ Transformatie klaar. {len(df_transformed)} rijen opgeslagen in {TRANSFORM_TABLE}.")
    except Exception as e:
        logger.error(f"❌ Fout tijdens transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == '__main__':
    transform_ned_pipeline()