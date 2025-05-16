#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - build_master_predictions - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_master_predictions")

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
        df_time = safe_load(conn, "dim_datetime")
        df_ned = safe_load(conn, "processed_NED_preds")
        df_weather = safe_load(conn, "process_weather_preds")
        df_now = safe_load(conn, "transform_meteo_forecast_now")

        df_time["target_datetime"] = pd.to_datetime(df_time["datetime"], utc=True)
        df_ned["validto"] = pd.to_datetime(df_ned["validto"], utc=True)
        df_ned["fetch_moment"] = pd.to_datetime(df_ned["fetch_moment"], utc=True)
        df_weather["target_datetime"] = pd.to_datetime(df_weather["target_datetime"], utc=True)

        if not df_now.empty:
            if "target_datetime" in df_now.columns:
                df_now["target_datetime"] = pd.to_datetime(df_now["target_datetime"], utc=True)
            elif "date" in df_now.columns:
                df_now = df_now.rename(columns={"date": "target_datetime"})
                df_now["target_datetime"] = pd.to_datetime(df_now["target_datetime"], utc=True)

        df_ned = df_ned.sort_values("fetch_moment").drop_duplicates("validto", keep="last")
        df_ned = df_ned.rename(columns={"validto": "target_datetime"})

        df = df_time.drop(columns=["datetime", "date"], errors="ignore")

        df = df.merge(df_ned, on="target_datetime", how="left")
        df = df.merge(df_weather, on="target_datetime", how="left", suffixes=("", "_weather"))

        if not df_now.empty:
            df = df.merge(df_now, on="target_datetime", how="left", suffixes=("", "_now"))

        # === Combineer suffix-kolommen naar originele naam ===
        suffix_sources = ["_weather", "_now"]
        base_cols = [col.replace("_weather", "").replace("_now", "") 
                     for col in df.columns if any(suffix in col for suffix in suffix_sources)]

        for col in set(base_cols):
            suffix_cols = [c for c in df.columns if c.startswith(col + "_")]
            if suffix_cols:
                for suffix_col in suffix_cols:
                    df[col] = df[col].combine_first(df[suffix_col])
                df = df.drop(columns=suffix_cols)

        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_values("target_datetime")

        logger.info(f"📊 Eindtabel: {df.shape[0]} rijen, {df.shape[1]} kolommen")
        logger.info(f"🧾 Kolommen: {df.columns.tolist()}")

        df.to_sql(MASTER_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ {MASTER_TABLE} succesvol opgeslagen")

    except Exception as e:
        logger.error(f"❌ Fout tijdens build: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    build_master()