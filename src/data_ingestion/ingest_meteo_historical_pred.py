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
CSV_PATH = PROJECT_ROOT / "src" / "data" / "weather" / "Weather_pred_past_3m_DeBilt.csv"

def create_hourly_params():
    """Create the full list of parameters including previous days for the API request"""
    base_variables = [
        "temperature_2m", 
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "snowfall", 
        "apparent_temperature",
        "diffuse_radiation", 
        "direct_normal_irradiance",
        "shortwave_radiation", 
        "direct_radiation"
    ]
    
    hourly_params = []
    
    # Add base variables
    hourly_params.extend(base_variables)
    
    # Add previous day versions
    for var in base_variables:
        for day in range(1, 8):  # previous_day1 through previous_day7
            hourly_params.append(f"{var}_previous_day{day}")
    
    logger.info(f"📊 Totaal aantal variabelen voor API: {len(hourly_params)}")
    return hourly_params

# === Open-Meteo API setup ===
now = datetime.now(timezone.utc)
now_date_str = now.strftime("%Y-%m-%d")

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Get the list of all required parameters
hourly_params = create_hourly_params()

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
    "latitude": 52.108499,
    "longitude": 5.180616,
    "hourly": hourly_params,
    "models": "knmi_seamless",
    "start_date": (now - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
    "end_date": now_date_str
}

def fetch_api_data():
    try:
        logger.info("🔄 API-data ophalen van Open-Meteo...")
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        logger.info(f"✅ API-data opgehaald voor {response.Latitude()}°N {response.Longitude()}°E")

        hourly = response.Hourly()
        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        data = {"date": timestamps}
        
        # Process each hourly variable
        for i, var in enumerate(params["hourly"]):
            try:
                data[var] = hourly.Variables(i).ValuesAsNumpy()
                logger.debug(f"📊 Variabele geladen: {var}")
            except Exception as e:
                logger.warning(f"⚠️ Fout bij laden van variabele {var}: {e}")
                data[var] = None  # Use None for missing data

        df = pd.DataFrame(data)
        logger.info(f"📈 API-data geladen: {len(df)} rijen, {len(df.columns)} kolommen")
        return df
    except Exception as e:
        logger.error(f"❌ Fout bij ophalen van API-data: {e}", exc_info=True)
        return pd.DataFrame()

def load_csv_backup():
    if not CSV_PATH.exists():
        logger.warning(f"⚠️ Geen CSV-backup gevonden op: {CSV_PATH}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(CSV_PATH)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info(f"📁 CSV-backup geladen ({len(df)} rijen, {len(df.columns)} kolommen)")
        
        # Check if CSV has the expected columns
        missing_cols = set(hourly_params) - set(df.columns)
        if missing_cols:
            logger.warning(f"⚠️ CSV mist {len(missing_cols)} kolommen die nodig zijn voor transformatie")
            logger.debug(f"Missende kolommen: {missing_cols}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Fout bij laden van CSV: {e}", exc_info=True)
        return pd.DataFrame()

def save_to_csv(df):
    """Save data to CSV as backup"""
    try:
        # Create directory if it doesn't exist
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = now.strftime("%Y-%m-%d")
        csv_path = CSV_PATH.with_name(f"Weather_pred_past_3m_DeBilt_{timestamp}.csv")
        
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Data opgeslagen als CSV-backup: {csv_path}")
    except Exception as e:
        logger.error(f"❌ Fout bij opslaan van CSV: {e}", exc_info=True)

def ingest():
    logger.info(f"📦 Ingest starten, schrijven naar {TABLE_NAME}")
    conn = sqlite3.connect(DB_PATH)

    try:
        # First try to get data from API
        df_api = fetch_api_data()
        
        # If API fails or returns empty data, try CSV backup
        if df_api.empty:
            logger.warning("⚠️ Geen data van API, proberen CSV-backup te laden...")
            df_combined = load_csv_backup()
        else:
            # If we have API data, merge with existing CSV data
            df_csv = load_csv_backup()
            
            if df_csv.empty:
                df_combined = df_api
                logger.info("ℹ️ Alleen API-data wordt gebruikt (geen CSV-backup)")
            else:
                # Combine API and CSV data, keeping the most recent values
                df_combined = pd.concat([df_csv, df_api])
                df_combined = df_combined.drop_duplicates(subset="date", keep="last")
                logger.info(f"🔄 Data gecombineerd: {len(df_combined)} unieke rijen")
            
            # Save combined data to CSV as backup
            save_to_csv(df_combined)
        
        if df_combined.empty:
            logger.error("❌ Geen data beschikbaar (API en CSV beiden leeg)")
            return
        
        # Sort by date and save to database
        df_combined = df_combined.sort_values("date")
        
        # Check for missing columns that might be needed for transformation
        expected_columns = set(["date"] + hourly_params)
        actual_columns = set(df_combined.columns)
        missing_columns = expected_columns - actual_columns
        
        if missing_columns:
            logger.warning(f"⚠️ Data mist {len(missing_columns)} kolommen die nodig zijn voor transformatie")
            logger.debug(f"Missende kolommen: {missing_columns}")
            
            # Add missing columns as NULL
            for col in missing_columns:
                df_combined[col] = None
        
        # Save to database
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