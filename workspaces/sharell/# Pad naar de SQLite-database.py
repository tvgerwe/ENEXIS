import pandas as pd
from entsoe import EntsoePandasClient
import time
import sqlite3

# Pad naar de SQLite-database
db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP.db'

# Maak verbinding met de database
conn = sqlite3.connect(db_path)

# Lees de 'raw_entsoe_obs'-tabel in een Pandas DataFrame
query = "SELECT * FROM transform_entsoe_obs"
df = pd.read_sql_query(query, conn)

# Sluit de verbinding
conn.close()

def summary(df):
    print("Dataset Summary:")
    print(df.describe(include='all'))  # Summary statistics
    print("\nMissing Values (NA) per Column:")
    print(df.isna().sum())  # Count of missing values

print(summary(df))