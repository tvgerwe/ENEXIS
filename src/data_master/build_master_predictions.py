#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - build_master_predictions - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_master_predictions")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
MASTER_TABLE = "master_predictions"

def safe_load(conn, table):
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        logger.info(f"✅ '{table}' geladen ({len(df)} rijen)")
        return df
    except Exception as e:
        logger.error(f"❌ Fout bij laden '{table}': {e}")
        return pd.DataFrame()

def build_master():
    logger.info(f"📦 Start build voor {MASTER_TABLE}")
    conn = sqlite3.connect(DB_PATH)

    try:
        df_ned = safe_load(conn, "processed_NED_preds")
        df_weather = safe_load(conn, "process_weather_preds")
        df_now = safe_load(conn, "transform_meteo_forecast_now")  # optioneel
        print(df_ned.columns.tolist())
        # Normaliseer kolomnamen
        if "target_datetime" in df_ned.columns:
            df_ned["target_datetime"] = pd.to_datetime(df_ned["target_datetime"], utc=True)
        elif "date" in df_ned.columns:
            df_ned = df_ned.rename(columns={"date": "target_datetime"})
            df_ned["target_datetime"] = pd.to_datetime(df_ned["target_datetime"], utc=True)
        else:
            raise KeyError("⚠️ processed_NED_preds mist kolom 'target_datetime' of 'date'")
        df_weather["target_datetime"] = pd.to_datetime(df_weather["target_datetime"], utc=True)
        if "target_datetime" in df_now.columns:
            df_now["target_datetime"] = pd.to_datetime(df_now["target_datetime"], utc=True)
        elif "date" in df_now.columns:
            df_now = df_now.rename(columns={"date": "target_datetime"})
            df_now["target_datetime"] = pd.to_datetime(df_now["target_datetime"], utc=True)

        # Start join op target_datetime
        df = df_ned.merge(df_weather, on="target_datetime", how="outer", suffixes=("", "_weather"))

        if not df_now.empty:
            df = df.merge(df_now, on="target_datetime", how="outer", suffixes=("", "_now"))

        # Drop dubbele kolommen of NaNs waar nodig
        df = df.dropna(subset=["target_datetime"])
        df = df.sort_values("target_datetime")

        logger.info(f"📊 Samengevoegd: {df.shape[0]} rijen, {df.shape[1]} kolommen")

        df.to_sql(MASTER_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ {MASTER_TABLE} opgeslagen")
    except Exception as e:
        logger.error(f"❌ Fout tijdens build: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    build_master()