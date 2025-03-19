pip install entsoe-py

import pandas as pd
from entsoe import EntsoePandasClient

import pandas as pd
from entsoe import EntsoePandasClient

# Insert your API key here once you get it
API_KEY = 'YOUR_API_KEY_HERE'

# Initialize ENTSO-E client
client = EntsoePandasClient(api_key=API_KEY)

# Define parameters
country_code = 'NL'
years = [2022, 2023, 2024]  # List of years to fetch
all_data = []  # To store yearly data

# Loop through each year and fetch data separately
for year in years:
    start = pd.Timestamp(f'{year}-01-01', tz='Europe/Amsterdam')
    end = pd.Timestamp(f'{year+1}-01-01', tz='Europe/Amsterdam')  # Exclusive end
    
    print(f"Fetching data for {year}...")
    yearly_data = client.query_load(country_code, start=start, end=end)
    all_data.append(yearly_data)

# Concatenate all years into one DataFrame
final_data = pd.concat(all_data)

# Save to CSV
final_data.to_csv('actual_total_load_nl_2022_2024.csv')

print("✅ Data successfully downloaded and saved as 'actual_total_load_nl_2022_2024.csv'")

