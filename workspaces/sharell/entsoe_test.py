import sqlite3
import pandas as pd

# Pad naar de SQLite-database
db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP.db'

# Maak verbinding met de database
conn = sqlite3.connect(db_path)

# Lees de 'raw_entsoe_obs'-tabel in een Pandas DataFrame
query = "SELECT * FROM transformed_entsoe_obs"
df = pd.read_sql_query(query, conn)

# Sluit de verbinding
conn.close()

print(df.head())




# Path to the SQLite database
db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP.db'

# Connect to the database
conn = sqlite3.connect(db_path)

# Query to fetch column names
query = "PRAGMA table_info(transformed_entsoe_obs)"
columns_info = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Print column names
print(columns_info['name'].tolist())

import sqlite3
import pandas as pd

# Path to the SQLite database
db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP.db'

# Connect to the database
conn = sqlite3.connect(db_path)

# Query to find the maximum timestamp
query = "SELECT MAX(Timestamp) AS max_timestamp FROM transformed_entsoe_obs"
max_timestamp = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Print the maximum timestamp
print("Maximum Timestamp:", max_timestamp['max_timestamp'].iloc[0])

import sqlite3
import pandas as pd

# Path to the SQLite database
db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP.db'

# Connect to the database
conn = sqlite3.connect(db_path)

# Query to list all tables in the database
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Print the table names
print("Tables in the database:", tables['name'].tolist())