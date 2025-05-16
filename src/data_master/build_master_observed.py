#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - build_master_observed - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_master_observed")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
MASTER_TABLE = "master_warp"

def safe_load_table(conn, table_name):
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        logger.info(f"✅ '{table_name}' geladen met {len(df)} rijen")
        return df
    except Exception as e:
        logger.error(f"❌ Kan '{table_name}' niet laden: {e}")
        return pd.DataFrame()

def build_master():
    logger.info(f"📦 Start build voor {MASTER_TABLE}")
    conn = sqlite3.connect(DB_PATH)

    try:
        df_time = safe_load_table(conn, "dim_datetime")
        df_entsoe = safe_load_table(conn, "transform_entsoe_obs")
        df_weather = safe_load_table(conn, "transform_weather_obs")
        df_ned = safe_load_table(conn, "transform_ned_obs_2")

        df_time["target_datetime"] = pd.to_datetime(df_time["datetime"], utc=True)
        df_entsoe["target_datetime"] = pd.to_datetime(df_entsoe["Timestamp"], utc=True)
        df_weather["target_datetime"] = pd.to_datetime(df_weather["date"], utc=True)
        df_ned["target_datetime"] = pd.to_datetime(df_ned["validto"], utc=True)

        df_entsoe = df_entsoe.drop(columns=["Timestamp"], errors="ignore")
        df_weather = df_weather.drop(columns=["date"], errors="ignore")
        df_ned = df_ned.drop(columns=["validto"], errors="ignore")

        df = df_time.drop(columns=["datetime", "date"], errors="ignore")
        df = df.merge(df_entsoe, on="target_datetime", how="left")
        df = df.merge(df_weather, on="target_datetime", how="left")
        df = df.merge(df_ned, on="target_datetime", how="left")

        df = df.fillna(0)
        logger.info(f"📊 Samengevoegd: {df.shape[0]} rijen, {df.shape[1]} kolommen")
        logger.info(f"🧾 Kolommen: {df.columns.tolist()}")

        df.to_sql(MASTER_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ {MASTER_TABLE} succesvol opgeslagen")
    except Exception as e:
        logger.error(f"❌ Fout bij bouwen van {MASTER_TABLE}: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    build_master()