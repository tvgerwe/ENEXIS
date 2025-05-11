import sqlite3
import pandas as pd

# Pad naar de SQLite-database
db_path = 'C:/Users/shba/Documents/ENEXIS/ENEXIS/src/data/WARP.db'

# Maak verbinding met de database
conn = sqlite3.connect(db_path)

# Lees de 'raw_entsoe_obs'-tabel in een Pandas DataFrame
query = "SELECT * FROM raw_entsoe_obs"
df = pd.read_sql_query(query, conn)

# Sluit de verbinding
conn.close()

print(df.head())