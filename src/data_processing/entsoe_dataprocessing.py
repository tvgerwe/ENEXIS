import pandas as pd
from entsoe import EntsoePandasClient
import time
import sqlite3
from pathlib import Path

# Pad naar de SQLite-database
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"

# Maak verbinding met de database
conn = sqlite3.connect(DB_PATH)

# Lees de 'raw_entsoe_obs'-tabel in een Pandas DataFrame
query = "SELECT * FROM raw_entsoe_obs"
df = pd.read_sql_query(query, conn)

# Sluit de verbinding
conn.close()

# DEBUG: Check original columns
print("Original columns:")
print(df.columns.tolist())

# Ensure 'Timestamp' column is datetime and use it as index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.set_index('Timestamp', inplace=True)

# Resample the data to hourly frequency and calculate the mean for each hour
df_hourly = df.resample('h').mean()
print("Data resampled to hourly frequency based on the past hour!")

# DEBUG: Check columns after resampling
print("Columns after resampling:")
print(df_hourly.columns.tolist())

# Shift timestamps by +1 hour (aligns with end-of-hour timestamps)
df_hourly.index = df_hourly.index + pd.Timedelta(hours=1)

# Convert price to kWh
df_hourly['Price'] = df_hourly['Price'] / 1000

# Add the timestamps as a column
df_hourly.reset_index(inplace=True)
print("Timestamps added to the final DataFrame!")

neighboring_countries = ['GB', 'NO']

# DEBUG: Check which flow columns exist
flow_columns = [col for col in df_hourly.columns if 'Flow' in col]
print("Available Flow columns:")
print(flow_columns)

# Safe processing of neighboring countries
for neighbor in neighboring_countries:
    from_col = f'Flow_{neighbor}_to_NL'
    to_col = f'Flow_NL_to_{neighbor}'
    
    if from_col in df_hourly.columns and to_col in df_hourly.columns:
        df_hourly[f'Flow_{neighbor}'] = df_hourly[from_col] - df_hourly[to_col]
        df_hourly.drop([from_col, to_col], axis=1, inplace=True)
        print(f"Processed flow for {neighbor}")
    else:
        print(f"Missing columns for {neighbor}: {from_col} exists: {from_col in df_hourly.columns}, {to_col} exists: {to_col in df_hourly.columns}")

# Only calculate total flow for countries that were successfully processed
processed_countries = [country for country in neighboring_countries 
                      if f'Flow_{country}' in df_hourly.columns]
if processed_countries:
    df_hourly['Total_Flow'] = sum(df_hourly[f'Flow_{country}'] for country in processed_countries)
else:
    print("No flow data processed - skipping Total_Flow calculation")

# Save the DataFrame to a CSV file
df_hourly.to_csv('transform_entsoe.csv', index=False)

# Connect to the SQLite database
DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
conn = sqlite3.connect(DB_PATH)

# Write the DataFrame to the database table 'transformed_entsoe_obs'
df_hourly.to_sql('transform_entsoe_obs', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data successfully written to database table 'transform_entsoe_obs'")