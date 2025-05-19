import pandas as pd
from entsoe import EntsoePandasClient
import time
import sqlite3

# Pad naar de SQLite-database

DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"

# Maak verbinding met de database
conn = sqlite3.connect(db_path)

# Lees de 'raw_entsoe_obs'-tabel in een Pandas DataFrame
query = "SELECT * FROM raw_entsoe_obs"
df = pd.read_sql_query(query, conn)

# Sluit de verbinding
conn.close()


# Ensure the index is datetime with UTC
#df.index = pd.to_datetime(df.index, utc=True) # You first try to convert the index before assigning it. This does nothing.
#print("Index converted to datetime with UTC!")

# Shift timestamps by 1 hour to calculate the mean for the past hour
df.index = df.index - pd.Timedelta(hours=1)


# Ensure the index is datetime with UTC
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index

# Shift timestamps by 1 hour to calculate the mean for the past hour
df.index = df.index - pd.Timedelta(hours=1)

# Resample the data to hourly frequency and calculate the mean for each hour

df_hourly = df.resample('h').mean()
print("Data resampled to hourly frequency based on the past hour!")

df_hourly['Price'] = df_hourly['Price'] / 1000  # Convert price to kWh

# Add the timestamps as a column
df_hourly.reset_index(inplace=True)
print("Timestamps added to the final DataFrame!")

# Shift timestamps by 1 hour to calculate the mean for the past hour
df.index = df.index - pd.Timedelta(hours=-1)

neighboring_countries = ['BE', 'DE', 'GB', 'DK', 'NO']

for neighbor in neighboring_countries:
    df_hourly[f'Flow_{neighbor}'] = df_hourly[f'Flow_{neighbor}_to_NL'] - df_hourly[f'Flow_NL_to_{neighbor}']

df_hourly['Total_Flow'] = sum(df_hourly[f'Flow_{n}'] for n in neighboring_countries)

# Save the DataFrame to a CSV file
df_hourly.to_csv('transform_entsoe.csv', index=False)

# Connect to the SQLite database
db_path = '../data/WARP.db'
conn = sqlite3.connect(db_path)

# Write the DataFrame to the database table 'transformed_entsoe_obs'
# If table exists, replace it. If not, create new table
df_hourly.to_sql('transform_entsoe_obs', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data successfully written to database table 'transform_entsoe_obs'")
