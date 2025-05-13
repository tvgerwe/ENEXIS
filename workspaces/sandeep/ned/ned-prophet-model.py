# env - model-run-tf_env Python 3.10.16

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator # To ensure demand axis are integer.

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from prophet import Prophet
import polars as pl

import os
from datetime import datetime
import time

import sqlite3

# Custom function for MAPE and sMAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# Function to compute AIC for regression models
def compute_aic(y_true, y_pred, num_params):
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    n = len(y_true)
    aic = n * np.log(mse) + 2 * num_params  # AIC formula
    return aic


# Connect to the SQLite database
db_path = '/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/WARP.db'
conn = sqlite3.connect(db_path)
# Connect to the SQLite database using the existing db_path
conn = sqlite3.connect(db_path)
# Step 2: Read data from table
df_pd_orig = pd.read_sql_query("SELECT * FROM master_warp ORDER BY datetime DESC", conn)
# Step 3: Close the connection
conn.close()

# Step 1: Convert 'validto' column to datetime
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
# Step 2: Sort the DataFrame by 'validto' to avoid data leakage
df = df_pd_orig.sort_values(by='datetime')

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['day_of_week'] = df['datetime'].dt.dayofweek
df['week_of_year'] = df['datetime'].dt.isocalendar().week

df['lag_1'] = df['Price'].shift(1)
df['lag_2'] = df['Price'].shift(2)
df['rolling_mean_3'] = df['Price'].shift(1).rolling(window=3).mean()

features = ['year', 'month', 'day', 'day_of_week', 'week_of_year', 'lag_1', 'lag_2', 'rolling_mean_3']
X = df[features]
y = df['Price']

tscv = TimeSeriesSplit(n_splits=5)

# Example to get the latest train-test split
for train_index, test_index in tscv.split(df):
    train = df.iloc[train_index].copy()
    test = df.iloc[test_index].copy()

# Step 4: Convert 'validto' (datetime) to numeric format (Unix timestamp in seconds)
train['datetime_numeric'] = train['datetime'].astype('int64') // 10**9  # Convert datetime to numeric timestamp
test['datetime_numeric'] = test['datetime'].astype('int64') // 10**9

# Step 1: Prepare training data for Prophet
train_prophet = train[['datetime', 'Price']].rename(columns={'datetime': 'ds', 'Price': 'y'})
test_prophet = test[['datetime', 'Price']].rename(columns={'datetime': 'ds', 'Price': 'y'})

# Remove timezone if present
train_prophet['ds'] = train_prophet['ds'].dt.tz_localize(None)
test_prophet['ds'] = test_prophet['ds'].dt.tz_localize(None)

model_run_start_time = time.time()

# Step 2: Train Prophet model
model = Prophet()
model.fit(train_prophet)

print("train complete")

# Step 6: Make predictions on the test set
X_test = test[['datetime_numeric']]
y_test = test['Price']

print("test start")
# Step 3: Create future dataframe for the test period
future = test_prophet[['ds']].copy()

# Step 4: Forecast
forecast = model.predict(future)
print("test complete")

# Step 5: Evaluation
y_true = test_prophet['y'].values
y_pred = forecast['yhat'].values

y_int_pred = np.round(future).astype(int)  # Rounds and converts to int

model_run_end_time = time.time()

# Calculate execution time in seconds
execution_time = model_run_end_time - model_run_start_time

# Step 6: Plot the forecast
model.plot(forecast)
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Value")
plt.tight_layout()
plt.show()

print("plot complete")


# Step 5: Evaluation
y_true = test_prophet['y'].values
y_pred = forecast['yhat'].values

y_int_pred = np.round(future).astype(int)  # Rounds and converts to int

# Step 6: Plot the forecast
model.plot(forecast)
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Value")
plt.tight_layout()
plt.show()

# Optional: Plot forecast components (trend, weekly, yearly seasonality)
model.plot_components(forecast)
plt.tight_layout()
plt.show()


# Step 7: Convert Predictions Back to Polars (Optional)
df_pred = pl.DataFrame({"X Values": X_test.values, "Actual": y_test.values, "Predicted": y_int_pred, "Diff": y_test.values - y_int_pred})
print(df_pred)

# Step 8: Visualize the results (optional)
plt.plot(test['datetime_numeric'], y_test, label='True Price')
plt.plot(test['datetime_numeric'], future, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# aic_prophet = compute_aic(y_test, y_int_pred, num_params=X_train.shape[1] + 1)



#residuals = y_test - y_int_pred
#mse = np.mean(residuals**2)
#n = len(y_test)
#aic_prophet = n * np.log(mse) + 2 * (train_prophet.shape[1] + 1)  # AIC formula


model_name = "Prophet"
print("model_name", "Prophet")
#print("mean_absolute_error", mean_absolute_error)
#print("mean_absolute_percentage_error", mean_absolute_percentage_error)
#print("symmetric_mape", symmetric_mape)
#print("aic_prophet", aic_prophet)
#print("execution_time", execution_time)

print("Model run complete")

