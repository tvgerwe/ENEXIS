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
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

import joblib

import polars as pl

import os
from datetime import datetime
import time

import sqlite3

from pathlib import Path
import logging
import json


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


CSV_DATA_DIR = config['ned']['ned_model_download_dir']

# Step 1: Read JSON data from a file
csv_file_path = os.path.join(CSV_DATA_DIR, f"warp-csv-dataset.csv")

with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

# Step 1: Convert 'validto' column to datetime
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
# Step 2: Sort the DataFrame by 'validto' to avoid data leakage
df = df_pd_orig.sort_values(by='datetime')


# Step 1: Initial datetime formatting
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)  # Ensure no timezone

# Step 2: Filter out negative prices and define target/features
# df = df[df['Price'] > 0].copy()  # Filter positive prices only
y = df[['datetime', 'Price']]    # Target
X = df.drop(columns=['Price'])   # Features (including datetime)

# Step 3: Define custom date ranges for training and testing
train_start = "2025-01-01"
train_end   = "2025-03-14"
test_start  = "2025-03-15"
test_end    = "2025-04-14"

# Step 4: Filter based on date ranges
X_train = X[(X['datetime'] >= train_start) & (X['datetime'] <= train_end)].copy()
X_test  = X[(X['datetime'] >= test_start) & (X['datetime'] <= test_end)].copy()

y_train = y[(y['datetime'] >= train_start) & (y['datetime'] <= train_end)].copy()
y_test  = y[(y['datetime'] >= test_start) & (y['datetime'] <= test_end)].copy()

# Step 5: Display training and testing ranges
print("Train Date Range:")
print(f"Start: {X_train['datetime'].min()}")
print(f"End:   {X_train['datetime'].max()}")
print("\nTest Date Range:")
print(f"Start: {X_test['datetime'].min()}")
print(f"End:   {X_test['datetime'].max()}")

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

# Step 6: Combine X and y for Prophet
# regressors = ['Total_Flow', 'Solar_Vol', 'temperature_2m']
regressors = ['month','shortwave_radiation','apparent_temperature','temperature_2m','direct_normal_irradiance','diffuse_radiation','yearday_sin',
              'Flow_BE','hour_sin','is_non_working_day','is_weekend,','is_holiday','weekday_cos','wind_speed_10m','hour_cos','weekday_sin',
              'cloud_cover','Flow_GB','Nuclear_Vol','yearday_cos','Flow_NO','Load']


# Sanity check: keep only regressors present in X_train
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
horizon = 30  # forecast days

'''
# Parameter grid
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_mode': ['additive', 'multiplicative'],
    'seasonality_prior_scale': [1.0, 10.0, 20.0]
}
'''

# Best Model Param grid
param_grid = {
    'changepoint_prior_scale': [0.1],
    'seasonality_mode': ['additive'],
    'seasonality_prior_scale': [1.0]
}


# Create list of all parameter combinations
import itertools
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

best_params = None
best_mae = float('inf')

# Step 4: Train Prophet model
model_run_start_time = time.time()

for params in all_params:
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        seasonality_prior_scale=params['seasonality_prior_scale']
    )
    
    model.fit(train_prophet)
    
    forecast = model.predict(test_prophet[['ds']])
    
    y_true = test_prophet['y'].values
    y_pred = forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)

    print(f"Params: {params} → MAE: {mae:.2f}")
    
    if mae < best_mae:
        best_mae = mae
        best_params = params

print("\n✅ Best Parameters Found:")
print(best_params)
print(f"Best MAE: {best_mae:.2f}")

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

# Step 11: Evaluation Metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
# mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, a_min=1e-10, a_max=None))) * 100
# smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

print("\n📊 Evaluation Metrics:")

r2 = r2_score(y_true, y_pred)

model_name = "Prophet"
print("model_name", "Prophet")
print(f"MAE   : {mae:.2f}")
print(f"MSE   : {mse:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.4f}")

# Define the filename
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

results = []
model_name = "Prophet"
comments = "Revised model with hyperparameter tuning and -ve price Run"
print("model_name", "Prophet")

model_execution_time = model_run_end_time - model_run_start_time

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

print(f"✅ Model evaluation saved to {model_results_file_path}")

model_file_path = f'{MODEL_RUN_RESULTS_DIR}prophet_hyper_tuned_model.pkl'

# Save model
joblib.dump(model, model_file_path)
print("Model saved.")

print(f"✅ Model evaluation saved to {model_results_file_path}")