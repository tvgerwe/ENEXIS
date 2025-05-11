import pandas as pd
from entsoe import EntsoePandasClient
import time

# Insert your API key here once you get it
API_KEY = '82aa28d4-59f3-4e3a-b144-6659aa9415b5'

# Initialize ENTSO-E client
client = EntsoePandasClient(api_key=API_KEY)

# Define parameters
country_code = 'NL'
neighboring_countries = ['BE', 'DE', 'GB', 'DK', 'NO']  # Pas dit aan op basis van de relevante buren
year = 2025  # List of years to fetch

# Data storage
all_data = []

# Function to fetch data with retries
def fetch_with_retries(fetch_func, *args, retries=3, delay=5, **kwargs):
    for attempt in range(retries):
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}, retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"Failed to fetch data after {retries} attempts")

# Loop through each year and fetch data separately

start = pd.Timestamp(f'{year}-01-01', tz='Europe/Amsterdam')
end = pd.Timestamp(f'{year+1}-01-01', tz='Europe/Amsterdam')  # Exclusive end

print(f"Fetching load data for {year}...")
yearly_load = fetch_with_retries(client.query_load, country_code, start=start, end=end).squeeze()  # Convert to 1D Series

print(f"Fetching load forecast for {year}...")
yearly_load_forecast = fetch_with_retries(client.query_load_forecast, country_code, start=start, end=end).squeeze()  # Convert to 1D Series

print(f"Fetching price data for {year}...")
yearly_price = fetch_with_retries(client.query_day_ahead_prices, country_code, start=start, end=end).squeeze()  # Convert to 1D Series

# Fetch cross-border flows
flow_data = {}
for neighbor in neighboring_countries:
    print(f"Fetching cross-border flow from {neighbor} to {country_code} for {year}...")
    yearly_flow_to = fetch_with_retries(client.query_crossborder_flows, country_code_from=neighbor, 
                                            country_code_to=country_code, start=start, end=end).squeeze()  # Convert to 1D Series
    flow_data[f'Flow_{neighbor}_to_{country_code}'] = yearly_flow_to

    print(f"Fetching cross-border flow from {country_code} to {neighbor} for {year}...")
    yearly_flow_from = fetch_with_retries(client.query_crossborder_flows, country_code_from=country_code, 
                                              country_code_to=neighbor, start=start, end=end).squeeze()  # Convert to 1D Series
    flow_data[f'Flow_{country_code}_to_{neighbor}'] = yearly_flow_from

# ...existing code...

# Merge all data
if not yearly_load.empty and not yearly_price.empty:
    # Align all Series to the same index (timestamps)
    common_index = yearly_load.index.union(yearly_price.index)
    for flow_series in flow_data.values():
        common_index = common_index.union(flow_series.index)

    # Reindex all Series to the common index
    yearly_load = yearly_load.reindex(common_index)
    yearly_price = yearly_price.reindex(common_index)
    for col_name in flow_data:
        flow_data[col_name] = flow_data[col_name].reindex(common_index)

    # Create a DataFrame with the aligned data
    df = pd.DataFrame({'Timestamp': common_index, 'Load': yearly_load.values, 'Price': yearly_price.values})
    
    # Add cross-border flow data to the DataFrame
    for col_name, flow_series in flow_data.items():
        df[col_name] = flow_series.values

    # Store yearly data
    all_data.append(df)
else:
    print(f"No data for year {year}")

# Concatenate all years into one DataFrame if there is data
if all_data:
    raw_entsoe = pd.concat(all_data)
else:
    raw_entsoe = pd.DataFrame()  # Create an empty DataFrame if no data is available

# Write the DataFrame to the database table 'raw_entsoe_obs'
if not raw_entsoe.empty:
    import sqlite3

    # Connect to the SQLite database
    db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP.db'
    conn = sqlite3.connect(db_path)

    # Write the DataFrame to the database table 'raw_entsoe_obs'
    # If table exists, replace it. If not, create new table
    raw_entsoe.to_sql('raw_entsoe_obs', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    print("Data successfully written to database table 'raw_entsoe_obs'")
else:
    print("No data available to write to the database.")