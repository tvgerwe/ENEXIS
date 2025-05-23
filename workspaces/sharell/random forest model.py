import sqlite3
import pandas as pd

# Pad naar de SQLite-database
db_path = 'C:/Users/shba/Documents/JADS project/ENEXIS/src/data/WARP2.db'

# Maak verbinding met de database
conn = sqlite3.connect(db_path)

# Lees de 'raw_entsoe_obs'-tabel in een Pandas DataFrame
query = "SELECT * FROM master_warp"
df = pd.read_sql_query(query, conn)

# Sluit de verbinding
conn.close()

from sklearn.ensemble import RandomForestRegressor

# Feature engineering
df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
df['dayofweek'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
df['Price_lag1'] = df['Price'].shift(1)
df['Price_lag24'] = df['Price'].shift(24)
df = df.dropna()

# Train/test split (laatste 20% als test)
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

X_train = train[['hour', 'dayofweek', 'Price_lag1', 'Price_lag24']]
y_train = train['Price']
X_test = test[['hour', 'dayofweek', 'Price_lag1', 'Price_lag24']]
y_test = test['Price']

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluatie
from sklearn.metrics import mean_absolute_error
print("MAE:", mean_absolute_error(y_test, y_pred))