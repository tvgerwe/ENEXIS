#!/usr/bin/env python3

import sqlite3
import pandas as pd
import logging
from pathlib import Path

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - master_warp - %(levelname)s - %(message)s'
)
logger = logging.getLogger("master_warp")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
MASTER_TABLE = "master_warp"

# === Helpers ===
def safe_load_table(conn, table_name):
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        logger.info(f"✅ '{table_name}' geladen met kolommen: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"❌ Kan tabel '{table_name}' niet laden: {e}")
        return pd.DataFrame()

def build_master_table():
    logger.info(f"📦 Using database: {DB_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"❌ Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        logger.info("📅 Laden van alle observatie-tabellen")
        df_time = safe_load_table(conn, "dim_datetime")
        df_entsoe = safe_load_table(conn, "transform_entsoe_obs")
        df_weather = safe_load_table(conn, "transform_weather_obs")
        df_ned = safe_load_table(conn, "transform_ned_obs_2")

        # Kolomnamen normaliseren
        df_entsoe.rename(columns={"Timestamp": "datetime"}, inplace=True)
        df_weather.rename(columns={"date": "datetime"}, inplace=True)
        df_ned.rename(columns={"validto": "datetime"}, inplace=True)

        for df_name, df in zip([
            "dim_datetime", "transform_entsoe_obs", "transform_weather_obs", "transform_ned_obs_2"
        ], [df_time, df_entsoe, df_weather, df_ned]):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            else:
                logger.warning(f"⚠️ Tabel '{df_name}' mist kolom 'datetime'.")

        logger.info("🔗 Joinen van observaties op datetime")
        df = df_time \
            .merge(df_entsoe, on="datetime", how="left") \
            .merge(df_weather, on="datetime", how="left") \
            .merge(df_ned, on="datetime", how="left")

        logger.info(f"✅ Gekoppeld resultaat: {df.shape[0]} rijen, {df.shape[1]} kolommen")

        logger.info(f"📊 Nulls vóór imputatie:\n{df.isna().sum()[df.isna().sum() > 0]}")

        df = df.fillna(0)

        logger.info(f"📆 Wegschrijven naar {MASTER_TABLE}")
        df.to_sql(MASTER_TABLE, conn, if_exists="replace", index=False)

        logger.info("🎉 master_warp succesvol opgebouwd en opgeslagen (alleen observaties).")
    except Exception as e:
        logger.error(f"❌ Fout bij bouwen van master_warp: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    build_master_table()
