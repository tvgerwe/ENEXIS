#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ingest_meteo_historical_pred - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingest_meteo_historical_pred")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
TABLE_NAME = "raw_weather_preds"
CSV_PATH = PROJECT_ROOT / "src" / "data_ingestion" / "historical_preds_archive.csv"

# === Open-Meteo API setup ===
now = datetime.now(timezone.utc)
now_date_str = now.strftime("%Y-%m-%d")

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
    "latitude": 52.108499,
    "longitude": 5.180616,
    "hourly": [
        "temperature_2m", "cloud_cover", "wind_speed_10m",
        "diffuse_radiation", "direct_normal_irradiance",
        "shortwave_radiation", "direct_radiation"
    ],
    "models": "knmi_seamless",
    "start_date": (now - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
    "end_date": now_date_str
}

def fetch_api_data():
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        logger.info("✅ API-data opgehaald")

        hourly = response.Hourly()
        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        data = {"date": timestamps}
        for i, var in enumerate(params["hourly"]):
            data[var] = hourly.Variables(i).ValuesAsNumpy()

        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"❌ Fout bij ophalen van API-data: {e}", exc_info=True)
        return pd.DataFrame()

def load_csv_backup():
    if not CSV_PATH.exists():
        logger.warning(f"⚠️ Geen CSV-backup gevonden op: {CSV_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    logger.info(f"📁 CSV-backup geladen ({len(df)} rijen)")
    return df

def ingest():
    logger.info(f"📦 Ingest starten, schrijven naar {TABLE_NAME}")
    conn = sqlite3.connect(DB_PATH)

    try:
        df_csv = load_csv_backup()
        df_api = fetch_api_data()

        df_combined = pd.concat([df_csv, df_api], axis=0)
        df_combined = df_combined.drop_duplicates(subset="date", keep="last")
        df_combined = df_combined.sort_values("date")

        df_combined.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        logger.info(f"✅ {TABLE_NAME} bevat nu {len(df_combined)} rijen")
        logger.info(f"📅 Datumbereik: {df_combined['date'].min()} t/m {df_combined['date'].max()}")
    except Exception as e:
        logger.error(f"❌ Fout tijdens ingest: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 DB-verbinding gesloten")

if __name__ == "__main__":
    ingest()
