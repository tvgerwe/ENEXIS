import pandas as pd
from entsoe import EntsoePandasClient
import sqlite3
import time
from pathlib import Path

# ─────────────────────────────────────────────
# 📍 Projectpad en databasedefinitie
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # gaat 2 niveaus omhoog vanuit dit bestand
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
OUTPUT_TABLE = "raw_entsoe_obs"
CSV_PATH = PROJECT_ROOT / "outputs" / f"{OUTPUT_TABLE}.csv"
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)  # maak outputmap aan als die niet bestaat

# ─────────────────────────────────────────────
# 🔑 API en parameters
# ─────────────────────────────────────────────
API_KEY = '82aa28d4-59f3-4e3a-b144-6659aa9415b5'
client = EntsoePandasClient(api_key=API_KEY)

country_code = 'NL'
neighboring_countries = ['GB', 'NO']
year = 2025  # pas aan indien nodig

# ─────────────────────────────────────────────
# 🔁 Retry-logica voor API-calls
# ─────────────────────────────────────────────
def fetch_with_retries(fetch_func, *args, retries=3, delay=5, **kwargs):
    for attempt in range(retries):
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}, retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"Failed after {retries} attempts")

# ─────────────────────────────────────────────
# 📥 Data ophalen
# ─────────────────────────────────────────────
start = pd.Timestamp(f"{year}-01-01", tz='Europe/Amsterdam')
end = pd.Timestamp(f"{year+1}-01-01", tz='Europe/Amsterdam')

print(f"Fetching ENTSO-E data for year {year}...")

yearly_load = fetch_with_retries(client.query_load, country_code, start=start, end=end).squeeze()
yearly_price = fetch_with_retries(client.query_day_ahead_prices, country_code, start=start, end=end).squeeze()
yearly_forecast = fetch_with_retries(client.query_load_forecast, country_code, start=start, end=end).squeeze()

flow_data = {}
for neighbor in neighboring_countries:
    flow_data[f'Flow_{neighbor}_to_{country_code}'] = fetch_with_retries(
        client.query_crossborder_flows, country_code_from=neighbor, country_code_to=country_code,
        start=start, end=end).squeeze()

    flow_data[f'Flow_{country_code}_to_{neighbor}'] = fetch_with_retries(
        client.query_crossborder_flows, country_code_from=country_code, country_code_to=neighbor,
        start=start, end=end).squeeze()

# ─────────────────────────────────────────────
# 🧱 Data structureren
# ─────────────────────────────────────────────
if not yearly_load.empty and not yearly_price.empty:
    all_index = yearly_load.index.union(yearly_price.index).union(yearly_forecast.index)
    for series in flow_data.values():
        all_index = all_index.union(series.index)

    df = pd.DataFrame({
        "Timestamp": all_index,
        "Load": yearly_load.reindex(all_index).values,
        "Price": yearly_price.reindex(all_index).values,
        "Forecast_Load": yearly_forecast.reindex(all_index).values,
    })

    for key, series in flow_data.items():
        df[key] = series.reindex(all_index).values

    df = df.sort_values("Timestamp")

    # Opslaan naar database
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        print(f"✅ Data opgeslagen in SQLite als tabel '{OUTPUT_TABLE}'")

else:
    print("⚠️ Geen bruikbare data beschikbaar om op te slaan.")