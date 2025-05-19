#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests_cache
from retry_requests import retry
from openmeteo_requests import Client

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ingest_meteo_forecast_now - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingest_meteo_forecast_now")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
TABLE_NAME = "raw_meteo_forecast_now"

# === Variabelen ophalen ===
VARS = [
    "temperature_2m", "wind_speed_10m", "apparent_temperature", "cloud_cover", 
    "snowfall", "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation"
]

def get_connection(db_path):
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database niet gevonden: {db_path}")
    return sqlite3.connect(db_path)

def fetch_forecast_now():
    start_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    end_date = (datetime.now(timezone.utc) + timedelta(days=7)).strftime('%Y-%m-%d')

    logger.info(f"🌦️ Ophalen van actuele forecast: {start_date} → {end_date}")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.12949,
        "longitude": 5.20514,
        "models": "knmi_seamless",
        "hourly": VARS,
        "start_date": start_date,
        "end_date": end_date
    }

    session = retry(requests_cache.CachedSession(".cache", expire_after=3600), retries=5)
    client = Client(session=session)
    responses = client.weather_api(url, params=params)

    response = responses[0]
    hourly = response.Hourly()

    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive='left'
    )

    data = {"date": timestamps}
    for i, var in enumerate(VARS):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    return df

def ingest_forecast_now():
    logger.info(f"📦 Verbinden met database: {DB_PATH}")
    conn = get_connection(DB_PATH)

    try:
        df = fetch_forecast_now()
        df["date"] = pd.to_datetime(df["date"])

        logger.info(f"💾 Forecast opslaan in tabel {TABLE_NAME} ({len(df)} rijen)")
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        logger.info("✅ Actuele forecast succesvol opgeslagen.")
    except Exception as e:
        logger.error(f"❌ Fout bij ophalen of opslaan: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    ingest_forecast_now()