#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path
import re

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
        df = pd.read_sql_query(f"SELECT * FROM {RAW_TABLE}", conn)
        logger.info(f"✅ {RAW_TABLE} geladen ({len(df)} rijen)")

        df["run_date"] = pd.to_datetime(df["date"], utc=True)
        df = df.drop(columns=["date"])

        long_rows = []
        pattern = re.compile(r"^(?P<variable>.+)_previous_day(?P<horizon>\d+)$")

        for col in df.columns:
            match = pattern.match(col)
            if match:
                variable = match.group("variable")
                horizon = int(match.group("horizon"))
                temp = df[["run_date", col]].copy()
                temp["variable"] = variable
                temp["value"] = temp[col]
                temp["horizon"] = horizon
                temp["target_datetime"] = temp["run_date"] - pd.to_timedelta(horizon, unit="D")
                temp = temp[["run_date", "target_datetime", "variable", "horizon", "value"]]
                long_rows.append(temp)

        df_long = pd.concat(long_rows, axis=0)
        df_long = df_long.dropna(subset=["value"])
        df_long = df_long.sort_values(["target_datetime", "variable"])

        logger.info(f"📊 Transformatie succesvol: {len(df_long)} rijen")
        df_long.to_sql(TRANSFORM_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ Weggeschreven naar {TRANSFORM_TABLE}")
    except Exception as e:
        logger.error(f"❌ Fout tijdens transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    transform()
