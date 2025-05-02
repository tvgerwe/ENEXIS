import pandas as pd
from entsoe import EntsoePandasClient
import time

# Load the existing CSV file
df = pd.read_csv('electricity_data_nl_2022_2025.csv', index_col=0, parse_dates=True)
print("CSV file loaded successfully!")

# Ensure the index is datetime with UTC
df.index = pd.to_datetime(df.index, utc=True)
print("Index converted to datetime with UTC!")

# Save the utc data to a new CSV file
df.to_csv('electricity_data_nl_2022_2025_utc.csv')

# Shift timestamps by 1 hour to calculate the mean for the past hour
df.index = df.index - pd.Timedelta(hours=1)

# Resample the data to hourly frequency and calculate the mean for each hour
df_hourly = df.resample('h').mean()
print("Data resampled to hourly frequency based on the past hour!")

# Save the resampled data to a new CSV file
df_hourly.to_csv('electricity_data_nl_2022_2025_hourly.csv')
print("Hourly data with flow saved successfully!")

neighboring_countries = ['BE', 'DE', 'GB', 'DK', 'NO']

for neighbor in neighboring_countries:
    df_hourly[f'Flow_{neighbor}'] = df_hourly[f'Flow_{neighbor}_to_NL'] - df_hourly[f'Flow_NL_to_{neighbor}']

df_hourly['Total_Flow'] = df_hourly['Flow_BE'] + df_hourly['Flow_DE'] + df_hourly['Flow_GB'] + df_hourly['Flow_DK'] + df_hourly['Flow_NO']
df_hourly.to_csv('electricity_data_nl_2022_2025_hourly_flow.csv')