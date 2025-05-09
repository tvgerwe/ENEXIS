#!/usr/bin/env python3

import os
import json
import requests
import pandas as pd
import sqlite3
import datetime
from retry_requests import retry
from openmeteo_requests.client import Client
import requests_cache

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = Client(session=retry_session)

# Define now and convert to YYYY-MM-DD format
now = datetime.datetime.now(datetime.timezone.utc)
now_date_str = now.strftime("%Y-%m-%d")

# Load configuration from JSON file
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

CONFIG_PATH = os.path.join(ROOT_DIR, 'workspaces', 'sandeep', 'config', 'api-call.json')
config = load_config(CONFIG_PATH)
endpoint = config['ned']['ned_api_endpoint']
headers = {
    'X-AUTH-TOKEN': config['ned']['demo-ned-api-key'],
    'Content-Type': 'application/json'
}

# Connect to the database and get most recent date
def get_connection(db_path):
    return sqlite3.connect(db_path)

WARP_DB_PATH = os.path.join(ROOT_DIR, 'src', 'data', 'WARP.db')
conn_data = get_connection(WARP_DB_PATH)
cursor = conn_data.cursor()
cursor.execute("SELECT MAX(validfrom) FROM some_historical_weather_table")
most_recent_date = cursor.fetchone()[0]

# Fetch data from the Open-Meteo API based on most recent date
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 52.12949,
    "longitude": 5.20514,
    "hourly": ["temperature_2m", "wind_speed_10m", "apparent_temperature", "cloud_cover", "snowfall", "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation"],
    "models": "knmi_seamless",
    "end_date": now_date_str,
    "start_date": most_recent_date,
}
responses = openmeteo.weather_api(url, params=params)

# Process single location
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data
hourly = response.Hourly()
variables = [
    "temperature_2m", "wind_speed_10m", "apparent_temperature", "cloud_cover", 
    "snowfall", "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation"
]
data = {}
for i, var in enumerate(variables):
    data[var] = hourly.Variables(i).ValuesAsNumpy()

# Create a DataFrame from the fetched data
recent_obs_dataframe = pd.DataFrame({
    'date': pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive='left'
    )
})
for i, var in enumerate(variables):
    recent_obs_dataframe[var] = data[var]

# Filter out future dates
recent_obs_dataframe = recent_obs_dataframe[recent_obs_dataframe['date'] <= now]
print(recent_obs_dataframe)

conn = sqlite3.connect(db_path)

# Read existing data from raw_weather_obs
existing_data = pd.read_sql_query("SELECT * FROM raw_weather_obs", conn)

# Convert date columns to datetime for both dataframes
existing_data['date'] = pd.to_datetime(existing_data['date'])
recent_obs_dataframe['date'] = pd.to_datetime(recent_obs_dataframe['date'])

# Remove any duplicates based on date and keep the latest values
merged_df = pd.concat([existing_data, recent_obs_dataframe])
initial_rows = len(merged_df)
merged_df = merged_df.drop_duplicates(subset='date', keep='last')
removed_rows = initial_rows - len(merged_df)
print(f"Removed {removed_rows} duplicate rows")

# Sort by date
merged_df = merged_df.sort_values('date')

# Write the merged dataframe back to the database
merged_df.to_sql('raw_weather_obs', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print(f"Data successfully merged. Total rows: {len(merged_df)}")
print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")