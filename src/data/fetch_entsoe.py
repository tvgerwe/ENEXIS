import pandas as pd
from entsoe import EntsoePandasClient
import time
from pathlib import Path

API_KEY = '82aa28d4-59f3-4e3a-b144-6659aa9415b5'
client = EntsoePandasClient(api_key=API_KEY)

def fetch_entsoe_data(years=[2022, 2023, 2024], save_dir="data/raw/") -> pd.DataFrame:
    country_code = "NL"
    neighboring_countries = ['BE', 'DE', 'GB', 'DK', 'NO']
    years = [2022, 2023, 2024]
    all_data = []

    def fetch_with_retries(fetch_func, *args, retries=3, delay=5, **kwargs):
        for attempt in range(retries):
            try:
                return fetch_func(*args, **kwargs)
            except Exception as e:
                print(f"Error: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to fetch data after {retries} attempts")

    for year in years:
        start = pd.Timestamp(f"{year}-01-01", tz="Europe/Amsterdam")
        end = pd.Timestamp(f"{year+1}-01-01", tz="Europe/Amsterdam")

        print(f"📥 Fetching data for {year}...")
        load = fetch_with_retries(client.query_load, country_code, start=start, end=end).squeeze()
        forecast = fetch_with_retries(client.query_load_forecast, country_code, start=start, end=end).squeeze()
        price = fetch_with_retries(client.query_day_ahead_prices, country_code, start=start, end=end).squeeze()

        flow_data = {}
        for neighbor in neighboring_countries:
            flow_data[f"Flow_{neighbor}_to_NL"] = fetch_with_retries(client.query_crossborder_flows, neighbor, country_code, start=start, end=end).squeeze()
            flow_data[f"Flow_NL_to_{neighbor}"] = fetch_with_retries(client.query_crossborder_flows, country_code, neighbor, start=start, end=end).squeeze()

        df = pd.DataFrame({"Load": load, "Forecast": forecast, "Price": price})
        for name, series in flow_data.items():
            df[name] = series

        all_data.append(df)

    final_df = pd.concat(all_data)
    final_df.index = final_df.index.tz_convert("UTC")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    raw_path = Path(save_dir) / "electricity_data_nl_2022_2024.csv"
    final_df.to_csv(raw_path)

    print(f"✅ Raw data saved to {raw_path}")

    # Resample + flow calc
    df_hourly = final_df.resample("H").mean()
    for neighbor in neighboring_countries:
        df_hourly[f"Flow_{neighbor}"] = df_hourly[f"Flow_{neighbor}_to_NL"] - df_hourly[f"Flow_NL_to_{neighbor}"]

    df_hourly["Total_Flow"] = df_hourly[[f"Flow_{n}" for n in neighboring_countries]].sum(axis=1)
    hourly_path = Path(save_dir) / "electricity_data_nl_2022_2024_hourly_flow.csv"
    df_hourly.to_csv(hourly_path)

    print(f"📊 Hourly flow data saved to {hourly_path}")
    return df_hourly
