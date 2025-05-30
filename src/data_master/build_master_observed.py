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


desired_order = [
    'Price', 'target_datetime', 'Load', 'shortwave_radiation', 'temperature_2m',
    'direct_normal_irradiance', 'diffuse_radiation', 'Flow_NO', 'yearday_cos',
    'Flow_GB', 'month', 'is_dst', 'yearday_sin','wind_speed_10m', 'is_non_working_day',
    'hour_cos', 'is_weekend', 'cloud_cover', 'weekday_sin', 'hour_sin',
    'weekday_cos', 'hour', 'day_of_week', 'day_of_year', 'is_holiday'
]

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
        df_weather = safe_load_table(conn, "transform_weather_obs")
        df_entsoe = safe_load_table(conn, "transform_entsoe_obs")

        df_time["target_datetime"] = pd.to_datetime(df_time["datetime"], utc=True)
        df_weather["target_datetime"] = pd.to_datetime(df_weather["date"], utc=True)
        df_entsoe["target_datetime"] = pd.to_datetime(df_entsoe["Timestamp"], utc=True)

        df_time = df_time.drop(columns=["datetime", "date"], errors="ignore")
        df_weather = df_weather.drop(columns=["date"], errors="ignore")
        df_entsoe = df_entsoe.drop(columns=["Timestamp"], errors="ignore")

        df = df_time.merge(df_entsoe, on="target_datetime", how="left")
        df = df.merge(df_weather, on="target_datetime", how="left")
        df = df.fillna(0)

        # Kolomfilter toepassen
        available_cols = [col for col in desired_order if col in df.columns]
        missing = [col for col in desired_order if col not in df.columns]
        if missing:
            logger.warning(f"⚠️ Ontbrekende kolommen in master_warp: {missing}")

        remaining_cols = [col for col in df.columns if col not in desired_order]
        df = df[available_cols + remaining_cols]
        df = df.sort_values("target_datetime")

        logger.info(f"📊 Eindtabel: {df.shape[0]} rijen, {df.shape[1]} kolommen")
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