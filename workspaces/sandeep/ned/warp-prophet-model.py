# env - enexis-may-03-env-run

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

"""
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
"""

CSV_DATA_DIR = config['ned']['ned_model_download_dir']

# Step 1: Read JSON data from a file
csv_file_path = os.path.join(CSV_DATA_DIR, f"warp-csv-dataset.csv")

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Step 1: Convert 'validto' column to datetime
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
# Step 2: Sort the DataFrame by 'validto' to avoid data leakage
df = df_pd_orig.sort_values(by='datetime')
# Step 3: Initial datetime formatting
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)  # Ensure no timezone

# df = df[df['Price'] > 0].copy()  # Filter positive prices only

# Step 4 - Define X and y variables
y = df[['datetime', 'Price']]    # Target
X = df.drop(columns=['Price'])   # Features (including datetime)

# Step 5: Define custom date ranges for training and testing
train_start = "2025-01-01"
train_end   = "2025-03-14"
test_start  = "2025-03-15"
test_end    = "2025-04-14"

# Step 6: Filter based on date ranges
X_train = X[(X['datetime'] >= train_start) & (X['datetime'] <= train_end)].copy()
X_test  = X[(X['datetime'] >= test_start) & (X['datetime'] <= test_end)].copy()

y_train = y[(y['datetime'] >= train_start) & (y['datetime'] <= train_end)].copy()
y_test  = y[(y['datetime'] >= test_start) & (y['datetime'] <= test_end)].copy()


# Display training and testing ranges
logger = logging.getLogger('Train Date Range:')
logger = logging.getLogger(f"Start: {X_train['datetime'].min()}")
logger = logging.getLogger(f"End:   {X_train['datetime'].max()}")
logger = logging.getLogger("\nTest Date Range:")
logger = logging.getLogger(f"Start: {X_test['datetime'].min()}")
logger = logging.getLogger(f"End:   {X_test['datetime'].max()}")

# print(X.columns.tolist())

"""
['datetime', 'hour', 'day_of_week', 'month', 'day_of_year', 'date', 'hour_sin', 'hour_cos', 
 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos', 'is_holiday', 'is_weekend', 
 'is_non_working_day', 'Load', 'Flow_BE_to_NL', 'Flow_NL_to_BE', 'Flow_DE_to_NL', 'Flow_NL_to_DE', 
 'Flow_GB_to_NL', 'Flow_NL_to_GB', 'Flow_DK_to_NL', 'Flow_NL_to_DK', 'Flow_NO_to_NL', 'Flow_NL_to_NO', 
 'Flow_BE', 'Flow_DE', 'Flow_GB', 'Flow_DK', 'Flow_NO', 'Total_Flow', 'temperature_2m', 
 'wind_speed_10m', 'apparent_temperature', 'cloud_cover', 'snowfall', 'diffuse_radiation', 
 'direct_normal_irradiance', 'shortwave_radiation', 'Wind_Vol', 'WindOffshore_Vol', 'Solar_Vol', 
 'Nuclear_Vol']

0 All, 1 Wind, 2 Solar, 3 Biogas, 4 HeatPump, 8 Cofiring, 9 Geothermal, 10 Other, 11 Waste, 12 BioOil, 13 Biomass
14 Wood, 17 WindOffshore, 18 FossilGasPower, 19 FossilHardCoal, 20 Nuclear, 21 WastePower, 22 WindOffshoreB, 23 NaturalGas, 24 Biomethane, 25 BiomassPower
26 OtherPower, 27 ElectricityMix, 28 GasMix, 31 GasDistribution, 35 CHP Total, 50 SolarThermal, 51 WindOffshoreC, 53 IndustrialConsumersGasCombination
54 IndustrialConsumersPowerGasCombination, 55 LocalDistributionCompaniesCombination, 56 AllConsumingGas
"""

# Step 7: Define regressors for Prophet
regressors = ['Total_Flow', 'Solar_Vol', 'temperature_2m']

# Step 8 Sanity check: keep only regressors present in X_train
available_regressors = [col for col in regressors if col in X_train.columns]

train_prophet = pd.concat(
    [y_train[['datetime', 'Price']].reset_index(drop=True), 
     X_train[available_regressors].reset_index(drop=True)],
    axis=1
)
test_prophet = pd.concat(
    [y_test[['datetime', 'Price']].reset_index(drop=True), 
     X_test[available_regressors].reset_index(drop=True)],
    axis=1
)

# Step 9 Providing ds and y columns
train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)

train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

print("\nProphet Training Data Sample:")
print(train_prophet.head())

print("y_train columns:", y_train.columns.tolist())
print("X_train columns (used regressors):", available_regressors)

model_run_start_time = time.time()

# Step 10: Train Prophet model with only available regressors

model = Prophet()

for reg in available_regressors:
    model.add_regressor(reg)

model.fit(train_prophet)
print("✅ Training complete")

#print(y_test.describe())
#print(y_test.head(10))
#print("Zero or negative actuals:")

# Ensure 'ds' and all regressors exist
#assert all(col in test_prophet.columns for col in ['ds', 'y'] + available_regressors)

# Predict directly
forecast = model.predict(test_prophet)

model_run_end_time = time.time()
print("✅ Forecast complete")

merged = test_prophet[['ds', 'y']].copy()
merged['yhat'] = forecast['yhat']

model_execution_time = model_run_end_time - model_run_start_time

# Step 2: Combine actual and predicted values
test_results = test_prophet[['ds', 'y']].copy()
test_results['yhat'] = forecast['yhat']  # add predictions

# Optional: include lower/upper uncertainty intervals
test_results['yhat_lower'] = forecast['yhat_lower']
test_results['yhat_upper'] = forecast['yhat_upper']
 
model_test_results_file_path = f'{MODEL_RUN_RESULTS_DIR}warp_test_predictions.csv'

# Step 3: Save to CSV
test_results.to_csv(model_test_results_file_path, index=False)

print("✅ Saved test predictions to model_test_results_file_path")

"""
for col in available_regressors:
    plt.figure(figsize=(10, 2))
    plt.plot(forecast['ds'], forecast[col])
    plt.title(f"{col} over forecast horizon")
    plt.show()
"""

# === Align predictions and actuals by date ===
forecast_indexed = forecast.set_index('ds')
test_prophet_indexed = test_prophet.set_index('ds')

# Inner join to keep only common dates and drop any rows with NaNs
merged = test_prophet_indexed[['y']].join(forecast_indexed[['yhat']], how='inner').dropna()

print(f"Aligned data length: {len(merged)}")
print(f"Date range: {merged.index.min()} to {merged.index.max()}")

# Calculate difference
diff = merged['y'] - merged['yhat']

# === Error metrics ===
mae = mean_absolute_error(merged['y'], merged['yhat'])
mse = mean_squared_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mse)
r2 = r2_score(merged['y'], merged['yhat'])

print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

# === Plot Prophet forecast ===
model.plot(forecast)
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.show()

"""
# === Plot Prophet components ===
model.plot_components(forecast)
plt.tight_layout()
plt.show()
"""

# Extract numpy arrays for plotting or further analysis
y_true = merged['y'].values
y_pred = merged['yhat'].values

# === Print execution time ===
execution_time = model_run_end_time - model_run_start_time
print(f"⏱️ Execution time: {execution_time:.2f} seconds")


# === Summary of evaluation metrics ===
print("\n📊 Evaluation Metrics:")
print(f"Model Name: Prophet")
comments = "Refactor code run 5 with three parameters"

"""
print(f"MAE   : {mae:.2f}")
print(f"MSE   : {mse:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.4f}")

print("Model run complete")
"""
# Define the filename
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Append results to results
results.append(["Prophet", mae, mse, rmse, r2, comments, model_execution_time, model_run_timestamp])

# Convert results to DataFrame
metrics_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2", "Comments", "Execution Time", "Run At"])

# Display Table
# print(metrics_df)

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