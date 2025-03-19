import pandas as pd
from entsoe import EntsoePandasClient

# Insert your API key here once you get it
API_KEY = 'b04cbae4-eff6-4077-99cd-e75b1a9433d2'

# Initialize ENTSO-E client
client = EntsoePandasClient(api_key=API_KEY)

# Define parameters
country_code = 'NL'
neighboring_countries = ['BE', 'DE', 'GB']  # Pas dit aan op basis van de relevante buren
years = [2022, 2023, 2024]  # List of years to fetch

# Data storage
all_data = []

# Loop through each year and fetch data separately
for year in years:
    start = pd.Timestamp(f'{year}-01-01', tz='Europe/Amsterdam')
    end = pd.Timestamp(f'{year+1}-01-01', tz='Europe/Amsterdam')  # Exclusive end

    print(f"Fetching load data for {year}...")
    yearly_load = client.query_load(country_code, start=start, end=end).squeeze()  # Convert to 1D Series
    
    print(f"Fetching price data for {year}...")
    yearly_price = client.query_day_ahead_prices(country_code, start=start, end=end).squeeze()  # Convert to 1D Series

    # Fetch cross-border flows
    flow_data = {}
    flow_data2 = {}
    for neighbor in neighboring_countries:
        print(f"Fetching cross-border flow from {neighbor} to {country_code} for {year}...")
        yearly_flow_in = client.query_crossborder_flows(country_code_from=neighbor, 
                                                     country_code_to=country_code, 
                                                     start=start, 
                                                     end=end).squeeze()  # Convert to 1D Series
        flow_data[f'Flow_{neighbor}_to_{country_code}'] = yearly_flow_in
        yearly_flow_out = client.query_crossborder_flows(country_code_from=country_code, 
                                                     country_code_to=neighbor, 
                                                     start=start, 
                                                     end=end).squeeze()  # Convert to 1D Series
        flow_data2[f'Flow_{country_code}_to_{neighbor}'] = yearly_flow_out

    # Merge all data
    df = pd.DataFrame({'Load': yearly_load, 'Price': yearly_price})
    for col_name, flow_series in flow_data.items():
        df[col_name] = flow_series
    for col_name, flow_series in flow_data2.items():
        df[col_name] = flow_series

    # Store yearly data
    all_data.append(df)

# Concatenate all years into one DataFrame
final_data = pd.concat(all_data)

# Save to CSV
final_data.to_csv('electricity_data_nl_2022_2024.csv')

print("Data saved successfully!")