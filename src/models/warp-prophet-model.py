# env - enexis-may-03-env-run
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
import itertools

logging.basicConfig(
    level=logging.INFO,
    filename=str(Path(__file__).parent / "logs" / "warp-prophet-model-json.log"),
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('warp-prophet-model')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.json"

# === Config Setup ===
if not CONFIG_PATH.exists():
    logger.error(f"❌ Config not found at : {CONFIG_PATH}")
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# === Model Run Results Directory ===
WARP_DATA_FILE_PATH = PROJECT_ROOT / "src" / "data" / "warp-csv-dataset.csv"
logger.info(f"WARP_DATA_FILE_PATH: {WARP_DATA_FILE_PATH}")

try:
    with open(WARP_DATA_FILE_PATH, 'rb') as csv_file:
        df_pd_orig = pd.read_csv(csv_file)
    logger.info(f"Loaded data from {WARP_DATA_FILE_PATH}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

results = []

# Ensure 'target_datetime' is parsed as timezone-naive datetime
df_pd_orig['datetime'] = df_pd_orig['target_datetime']
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
df = df_pd_orig.sort_values(by='datetime')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)

# Define X and y
y = df[['datetime', 'Price']]
X = df.drop(columns=['Price'])

# Ensure 'datetime' is timezone-naive
train_start = "2025-01-01"
train_end   = "2025-03-07"
test_start  = "2025-03-08"
test_end    = "2025-03-14"

# Split the data into training and testing sets based on datetime
X_train = X[(X['datetime'] >= train_start) & (X['datetime'] <= train_end)].copy()
X_test  = X[(X['datetime'] >= test_start) & (X['datetime'] <= test_end)].copy()
y_train = y[(y['datetime'] >= train_start) & (y['datetime'] <= train_end)].copy()
y_test  = y[(y['datetime'] >= test_start) & (y['datetime'] <= test_end)].copy()

logger.info(f"Train Date Range: Start: {X_train['datetime'].min()} End: {X_train['datetime'].max()}")
logger.info(f"Test Date Range: Start: {X_test['datetime'].min()} End: {X_test['datetime'].max()}")

# First run with simple model only

model_run_start_time = time.time()

# Combine datetime and Price only (no regressors)
train_prophet = y_train[['datetime', 'Price']].reset_index(drop=True)
test_prophet = y_test[['datetime', 'Price']].reset_index(drop=True)

# Rename columns for Prophet compatibility
train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)

# Ensure datetime is timezone naive
train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

# Create and fit Prophet model (without regressors)
model = Prophet()
model.fit(train_prophet)

forecast = model.predict(test_prophet)
y_true = test_prophet['y'].values
y_pred = forecast['yhat'].values

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
logger.info(f"RMSE: {rmse:.2f}")

model_run_end_time = time.time()
logger.info("✅ Base Model Train complete with RMSE: {:.2f}".format(rmse))

forecast_indexed = forecast.set_index('ds')
test_prophet_indexed = test_prophet.set_index('ds')
merged = test_prophet_indexed[['y']].join(forecast_indexed[['yhat']], how='inner').dropna()

logger.info(f"Aligned data length: {len(merged)}")
logger.info(f"Date range: {merged.index.min()} to {merged.index.max()}")

diff = merged['y'] - merged['yhat']
mse = mean_squared_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mse)
logger.info("✅ Base Model test complete with RMSE: {:.2f}".format(rmse))

#model.plot(forecast)
#plt.title("Prophet Train Forecast")
#plt.xlabel("Date")
#plt.ylabel("Predicted Price")
#plt.tight_layout()
# plt.show()

#model.plot_components(forecast)
#plt.tight_layout()
# plt.show()

#y_true = merged['y'].values
#y_pred = merged['yhat'].values
execution_time = model_run_end_time - model_run_start_time
logger.info(f"Execution time: {execution_time:.2f} seconds")

logger.info("\n📊 Evaluation Metrics for base model:")
logger.info(f"Model Name: Prophet")
logger.info(f"RMSE: {rmse:.3f}")

comments = "Base model run on 29th May with 10 40 AM run"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", rmse, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "R2", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv"


# Run with lag and horizon

from datetime import timedelta

model_run_start_time = time.time()

# Parameters
lag = timedelta(hours=36)                     # Delay after last training point
forecast_horizon = timedelta(hours=144)       # Forecast window size (6 days)

# Prepare training data (no regressors)
train_prophet = y_train[['datetime', 'Price']].reset_index(drop=True)
train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)

# Fit the model
model = Prophet()
model.fit(train_prophet)

# Calculate forecast start and end based on lag and horizon
forecast_start = train_prophet['ds'].max() + lag
forecast_end = forecast_start + forecast_horizon
forecast_days = (forecast_end - forecast_start).days + 1

# Create future dataframe (with lag applied)
future = model.make_future_dataframe(periods=forecast_days, freq='D')

# Filter to lag + horizon range only
future = future[future['ds'] >= forecast_start]

# Predict
forecast = model.predict(future)

# Evaluate against test set if test data overlaps
test_prophet = y_test[['datetime', 'Price']].reset_index(drop=True)
test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

# Align and evaluate
forecast_indexed = forecast.set_index('ds')
test_prophet_indexed = test_prophet.set_index('ds')
merged = test_prophet_indexed[['y']].join(forecast_indexed[['yhat']], how='inner').dropna()

if merged.empty:
    logger.warning("⚠️ No overlapping dates between forecast and test set!")
else:
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    logger.info(f"✅ Forecast Horizon: {forecast_start.date()} → {forecast_end.date()}")
    logger.info(f"Aligned data points: {len(merged)}")
    logger.info(f"Lagged forecast RMSE: {rmse:.3f}")

model_run_end_time = time.time()
logger.info("✅ Lagged Forecast Complete")


######

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta

# -------------------- SETUP --------------------

# Define regressors
regressors = [
    'Load','shortwave_radiation','temperature_2m','direct_normal_irradiance','diffuse_radiation',
    'Flow_NO','yearday_cos','Flow_GB','month','is_dst','yearday_sin','is_non_working_day',
    'hour_cos','is_weekend','cloud_cover','weekday_sin','hour_sin','weekday_cos'
]


available_regressors = [col for col in X_train.columns if col in regressors]

# Parameter grid
# Reduced hyperparameter grid for faster grid search
param_grid = {
    'seasonality_mode': ['additive','multiplicative'],
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
    'holidays_prior_scale': [0.1, 1.0],
    'changepoint_range': [0.8],
    'n_changepoints': [25, 50]
}
"""
param_grid = {
    'changepoint_prior_scale': [0.5],
    'seasonality_mode': ['additive'],
    'seasonality_prior_scale': [1.0],
    'holidays_prior_scale': [1.0],
    'changepoint_range': [0.8]
}
"""

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# -------------------- HYPERPARAMETER TUNING --------------------

best_rmse = float('inf')
best_model = None
best_forecast = None
best_params = None

train_prophet = pd.concat([
    y_train[['datetime', 'Price']].reset_index(drop=True),
    X_train[available_regressors].reset_index(drop=True)
], axis=1)
test_prophet = pd.concat([
    y_test[['datetime', 'Price']].reset_index(drop=True),
    X_test[available_regressors].reset_index(drop=True)
], axis=1)

train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

for params in all_params:
    model = Prophet(**params)
    for reg in available_regressors:
        model.add_regressor(reg)
    model.fit(train_prophet)
    
    forecast = model.predict(test_prophet)
    y_true = test_prophet['y'].values
    y_pred = forecast['yhat'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_forecast = forecast
        best_params = params

logger.info(f"✅ Best Parameters: {best_params}")
logger.info(f"✅ Best RMSE: {best_rmse:.3f}")

# -------------------- SAVE MODEL --------------------

model_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "prophet_hyper_tuned_model.pkl"
joblib.dump(best_model, model_file_path)

# -------------------- ROLLING FORECAST for training and Test window --------------------

window_size = 30
test_size = 7
step_size = 1
start_date = pd.Timestamp('2025-03-15')
end_date = pd.Timestamp.today() - timedelta(days=window_size + test_size)

all_preds, all_actuals, all_timestamps, all_horizons = [], [], [], []
window_rmses = []
current_start = start_date

while current_start <= end_date:
    train_start = current_start
    train_end = train_start + timedelta(days=window_size - 1, hours=23)
    test_start = train_end + timedelta(hours=1)
    test_end = test_start + timedelta(days=test_size - 1, hours=23)

    train = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)]
    test = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)]

    if len(train) == 0 or len(test) == 0:
        current_start += timedelta(days=step_size)
        continue

    y_train_roll = train[['datetime', 'Price']]
    X_train_roll = train[available_regressors]
    y_test_roll = test[['datetime', 'Price']]
    X_test_roll = test[available_regressors]

    train_prophet = pd.concat([y_train_roll.reset_index(drop=True), X_train_roll.reset_index(drop=True)], axis=1)
    test_prophet = pd.concat([y_test_roll.reset_index(drop=True), X_test_roll.reset_index(drop=True)], axis=1)

    train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
    test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
    train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
    test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)

    model = Prophet(**best_params)
    for reg in available_regressors:
        model.add_regressor(reg)
    model.fit(train_prophet)

    forecast = model.predict(test_prophet)
    y_true = test_prophet['y'].values
    y_pred = forecast['yhat'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    window_rmses.append(rmse)

    horizons = ((test_prophet['ds'] - test_start).dt.days).values
    all_preds.extend(y_pred)
    all_actuals.extend(y_true)
    all_timestamps.extend(test_prophet['ds'].values)
    all_horizons.extend(horizons)

    current_start += timedelta(days=step_size)

avg_rmse = np.mean(window_rmses)
logger.info(f"📊 Average RMSE across windows: {avg_rmse:.3f}")

# Final evaluation
assert len(all_preds) == len(all_actuals) == len(all_timestamps) == len(all_horizons)
final_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
final_mae = mean_absolute_error(all_actuals, all_preds)
final_r2 = r2_score(all_actuals, all_preds)

logger.info(f"🎯 Rolling RMSE: {final_rmse:.3f} | MAE: {final_mae:.3f} | R2: {final_r2:.3f}")

# Convert all lists to numpy arrays for consistency

# Now, build a DataFrame with the horizon included
rolling_window_result_df = pd.DataFrame({
    'Timestamp': all_timestamps,
    'Actual': all_actuals,
    'Predicted': all_preds,
    'Horizon': all_horizons
})

print(rolling_window_result_df.head())

# Pivot so each horizon is a column
pivot_df = rolling_window_result_df.pivot_table(index='Timestamp', columns='Horizon', values='Predicted', aggfunc='first')
pivot_df.columns = [f'Predicted_{int(h+1)}d_ahead' for h in pivot_df.columns]  # 1-based
pivot_df = pivot_df.reset_index()

print(pivot_df.head())

# Add actuals
actuals = rolling_window_result_df.drop_duplicates('Timestamp')[['Timestamp', 'Actual']]
pivot_df = pd.merge(pivot_df, actuals, on='Timestamp', how='left')

print(pivot_df.head())

# Gemiddelde RMSE, MAE en feature importance
print(f"\nAverage RMSE: {np.mean(final_rmse):.2f}")

# Convert to DataFrame for easier horizon slicing
df_eval = pd.DataFrame({
    'timestamp': all_timestamps,
    'actual': all_actuals,
    'predicted': all_preds,
    'horizon': all_horizons
})

# Ensure proper types
df_eval['horizon'] = df_eval['horizon'].astype(int)

# Calculate RMSE per horizon
horizon_rmse_log = {}
max_horizon = df_eval['horizon'].max()

logger.info("📉 Horizon-wise RMSEs:")

for h in range(max_horizon + 1):  # horizon=0 means 1-step ahead
    df_h = df_eval[df_eval['horizon'] == h]
    if len(df_h) == 0:
        continue
    rmse_h = np.sqrt(mean_squared_error(df_h['actual'], df_h['predicted']))
    logger.info(f"🔹 Horizon {h+1}-step ahead → RMSE: {rmse_h:.4f}")
    horizon_rmse_log[f"rmse_horizon_{h+1}"] = rmse_h

horizon_rmse_df = pd.DataFrame(list(horizon_rmse_log.items()), columns=["Horizon", "RMSE"])
horizon_model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "horizon_wise_rmse.csv"
horizon_rmse_df.to_csv(horizon_model_results_file_path, index=False)
logger.info("✅ Horizon-wise RMSEs saved to horizon_wise_rmse.csv")

