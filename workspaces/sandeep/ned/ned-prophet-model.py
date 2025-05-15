# env - model-run-tf_env Python 3.10.16

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator # To ensure demand axis are integer.

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prophet import Prophet
import polars as pl

import os
from datetime import datetime
import time

import sqlite3

from pathlib import Path
import logging
import json
import joblib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_run_prophet')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

# === CONFIG ===

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_RUN_RESULTS_DIR = config['ned']['ned_model_download_dir']

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

results = []

# Connect to the SQLite database
db_path = '/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/WARP.db'
conn = sqlite3.connect(db_path)
# Connect to the SQLite database using the existing db_path
conn = sqlite3.connect(db_path)
# Step 2: Read data from table
df_pd_orig = pd.read_sql_query("SELECT * FROM master_warp ORDER BY datetime DESC", conn)

# df_pd_orig = pd.read_sql_query("SELECT * FROM raw_entsoe_obs ORDER BY Timestamp DESC", conn)
#df_pd_orig["datetime"] = df_pd_orig["Timestamp"]
# Step 3: Close the connection
conn.close()

# Step 1: Convert 'validto' column to datetime
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
# Step 2: Sort the DataFrame by 'validto' to avoid data leakage
df = df_pd_orig.sort_values(by='datetime')

# Step 1: Filter out negative Price and prepare data
# df = df[df['Price'] > 0].copy()                   # Keep only rows with positive Price
df['datetime'] = pd.to_datetime(df['datetime'])   # Ensure datetime column is datetime type
df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone info if present

# Step 2: Define target and features
y = df[['datetime', 'Price']]                     # Target with timestamp
X = df.drop(columns=['Price'])                    # All other features, including datetime

# Step 3: Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Print train/test datetime range
print("Train Date Range:")
print(f"Start: {X_train['datetime'].min()}")
print(f"End:   {X_train['datetime'].max()}")

print("\nTest Date Range:")
print(f"Start: {X_test['datetime'].min()}")
print(f"End:   {X_test['datetime'].max()}")

# Step 4: Merge X and y for Prophet input
train_prophet = pd.concat([y_train.reset_index(drop=True), 
                           X_train.drop(columns=['datetime']).reset_index(drop=True)], axis=1)
test_prophet = pd.concat([y_test.reset_index(drop=True), 
                          X_test.drop(columns=['datetime']).reset_index(drop=True)], axis=1)

# Step 5: Rename for Prophet
train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)

# Step 6: Ensure datetime has no timezone
train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

# (Optional) Show the head of the training DataFrame for verification
print("\nProphet Training Data:")
print(train_prophet.head())


# Step 4: Train Prophet model
model_run_start_time = time.time()
model = Prophet()
model.fit(train_prophet)
print("✅ Training complete")

print(y_test.describe())
print(y_test.head(10))
print("Zero or negative actuals:")
#print(y_test[y_test <= 0])


# Step 5: Forecasting
future = test_prophet[['ds']].copy()
forecast = model.predict(future)
print("✅ Forecast complete")
model_run_end_time = time.time()


# Step 6: Evaluation
y_true = test_prophet['y'].values
y_pred = forecast['yhat'].values
diff = y_true - y_pred


# Step 7: Plot Forecast
model.plot(forecast)
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.show()

# Step 8: Plot Forecast Components
model.plot_components(forecast)
plt.tight_layout()
plt.show()

# Step 9: Create Polars DataFrame for comparison
x_values = X_test['validto'].to_numpy() if 'validto' in X_test.columns else X_test.index.to_numpy()
df_pred = pl.DataFrame({
    "X_Values": x_values,
    "Actual": y_true,
    "Predicted": y_pred,
    "Diff": diff
})

# Step 10: Optional custom visualization
plt.figure(figsize=(10, 5))
plt.plot(test_prophet['ds'], y_true, label='Actual Price')
plt.plot(test_prophet['ds'], y_pred, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price')
plt.legend()
plt.tight_layout()
plt.show()


# Final: Print execution time
print(f"⏱️ Execution time: {model_run_end_time - model_run_start_time:.2f} seconds")
model_execution_time = model_run_end_time - model_run_start_time

# Step 11: Evaluation Metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
# mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, a_min=1e-10, a_max=None))) * 100
# smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

print("\n📊 Evaluation Metrics:")

r2 = r2_score(y_true, y_pred)

model_name = "Prophet"
comments = "Saving the Model Run"
print("model_name", "Prophet")
print(f"MAE   : {mae:.2f}")
print(f"MSE   : {mse:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.4f}")

# print(f"MAPE  : {mape:.2f}%")
# print(f"sMAPE : {smape:.2f}%")

print("Model run complete")

# Define the filename
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


# Append results to results
results.append(["Prophet", mae, mse, rmse, r2, comments, model_execution_time, model_run_timestamp])

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2", "Comments", "Execution Time", "Run At"])

# Display Table
print(metrics_df)


model_results_file_path = f'{MODEL_RUN_RESULTS_DIR}warp-prophet-model-results.csv'

# Step 10: Check if file exists, then append or create
if os.path.exists(model_results_file_path):
    # Append to existing file
    existing_results = pd.read_csv(model_results_file_path)
    updated_results = pd.concat([existing_results, metrics_df], ignore_index=True)
    updated_results.to_csv(model_results_file_path, index=False)
else:
    # Create new file
    metrics_df.to_csv(model_results_file_path, index=False)

model_file_path = f'{MODEL_RUN_RESULTS_DIR}prophet_model.pkl'

# Save model
joblib.dump(model, model_file_path)
print("Model saved.")

print(f"✅ Model evaluation saved to {model_results_file_path}")