#!/usr/bin/env python3

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
import requests_cache
from retry_requests import retry
from openmeteo_requests import Client

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ingest_meteo_obs - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingest_meteo_obs")

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
TABLE_NAME = "raw_meteo_obs_test"

# === Open-Meteo variabelen (KNMI Seamless) ===
VARS = [
    "temperature_2m", "wind_speed_10m", "apparent_temperature", "cloud_cover", 
    "snowfall", "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation"
]

def get_connection(db_path):
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database niet gevonden: {db_path}")
    return sqlite3.connect(db_path)

def get_last_observed_date(conn):
    try:
        df = pd.read_sql_query(f"SELECT MAX(date) as max_date FROM {TABLE_NAME}", conn)
        if pd.notnull(df.at[0, 'max_date']):
            return pd.to_datetime(df.at[0, 'max_date']).strftime('%Y-%m-%d')
    except:
        pass
    return "2025-01-01"

def fetch_openmeteo_obs(start_date, end_date):
    logger.info(f"🌦️ Ophalen van observaties: {start_date} → {end_date}")

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
    df = df[df["date"] <= datetime.now(timezone.utc)]
    return df

def ingest_meteo_obs():
    logger.info(f"📦 Verbinden met database: {DB_PATH}")
    conn = get_connection(DB_PATH)

    try:
        start_date = get_last_observed_date(conn)
        end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        df_new = fetch_openmeteo_obs(start_date, end_date)

        # Bestaande data ophalen
        try:
            df_existing = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
            df_existing["date"] = pd.to_datetime(df_existing["date"])
        except:
            df_existing = pd.DataFrame(columns=df_new.columns)

        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined.drop_duplicates(subset="date", keep="last")
        df_combined = df_combined.sort_values("date")

        logger.info(f"💾 Opslaan in tabel: {TABLE_NAME} ({len(df_combined)} rijen totaal)")
        df_combined.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        logger.info("✅ Observaties succesvol opgeslagen.")
    except Exception as e:
        logger.error(f"❌ Fout bij inladen observaties: {e}", exc_info=True)
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")

if __name__ == "__main__":
    ingest_meteo_obs()