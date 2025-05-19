#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path
import re
from functools import reduce

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

        df["target_datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.drop(columns=["date"])

        pattern = re.compile(r"^(?P<varname>.+)_previous_day(?P<day>\d+)$")
        variable_map = {}

        for col in df.columns:
            match = pattern.match(col)
            if match:
                varname = match.group("varname")
                day = int(match.group("day"))
                variable_map.setdefault(varname, []).append((day, col))

        result_frames = []

        for var, entries in variable_map.items():
            rows = []
            for horizon, col in sorted(entries):
                temp = df[["target_datetime", col]].copy()
                temp["run_date"] = temp["target_datetime"] - pd.Timedelta(days=horizon)
                temp.rename(columns={col: var}, inplace=True)
                rows.append(temp)

            merged = pd.concat(rows)
            result_frames.append(merged)

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

        df_final = df_final.dropna(how="all", subset=[col for col in df_final.columns if col not in ["run_date", "target_datetime"]])
        df_final = df_final.sort_values(["target_datetime", "run_date"])

        logger.info(f"📊 Transformatie klaar: {df_final.shape[0]} rijen, {df_final.shape[1]} kolommen")
        df_final.to_sql(TRANSFORM_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ Weggeschreven naar {TRANSFORM_TABLE}")
    except Exception as e:
        logger.error(f"❌ Fout tijdens transformatie: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    transform()