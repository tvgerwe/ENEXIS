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
import sys

# Zorg dat build_training_set geïmporteerd is
current_dir = Path.cwd()
while current_dir.name != "ENEXIS" and current_dir.parent != current_dir:
    current_dir = current_dir.parent
project_root = current_dir
utils_path = project_root / "src" / "utils"
sys.path.append(str(utils_path))
from build_training_set import build_training_set

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

FEATURES = [
    'Load', 'shortwave_radiation', 'temperature_2m', 
    'Flow_NO', 'yearday_cos', 'Flow_GB', 
    'yearday_sin', 
    'hour_sin']
target = 'Price'

# wind_speed_10m

# Initial training window
base_start = "2025-01-01 00:00:00"
base_end = "2025-03-14 23:00:00"
base_run = "2025-03-15 00:00:00"

rmse_results = []
results = []



print("🔍 Testing Prophet Model Top 10 features - RMSE per forecast day")
print("=" * 60)

model_run_start_time = time.time()

for i in range(30):
    start = pd.Timestamp(base_start) + pd.Timedelta(days=i)
    end = pd.Timestamp(base_end) + pd.Timedelta(days=i)
    run_date = pd.Timestamp(base_run) + pd.Timedelta(days=i)

    try:
        df = build_training_set(
            train_start=start.strftime("%Y-%m-%d %H:%M:%S"),
            train_end=end.strftime("%Y-%m-%d %H:%M:%S"),
            run_date=run_date.strftime("%Y-%m-%d %H:%M:%S")
        )

        if df is None or df.empty:
            print(f"Day {i+1}: ❌ No training data returned")
            continue

        df['target_datetime'] = pd.to_datetime(df['target_datetime'], utc=True)
        df = df.sort_values('target_datetime')

        run_date_utc = run_date.tz_localize("UTC")

        # Split into training and testing sets
        train_data = df[df['target_datetime'] <= run_date_utc]
        test_data = df[df['target_datetime'] > run_date_utc]

        # Drop any missing data in training and test
        train_data = train_data.dropna(subset=['target_datetime', target] + FEATURES)
        test_data = test_data.dropna(subset=['target_datetime', target] + FEATURES)

        if test_data.empty or train_data.empty:
            print(f"Day {i+1}: ❌ Not enough data for training or testing")
            continue

        # Prepare data for Prophet
        prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + FEATURES]
        prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
        prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + FEATURES]
        prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

        # Train Prophet model with extra regressors
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        for reg in FEATURES:
            model.add_regressor(reg)
        model.fit(prophet_train)

        # Forecast for the test period
        future = prophet_test[['ds'] + FEATURES]
        forecast = model.predict(future)
        y_pred = forecast['yhat'].values
        y_test = prophet_test['y'].values

        # Sla de eerste 24 uur over
        if len(y_pred) > 24:
            y_pred = y_pred[24:]
            y_test = y_test[24:]
        else:
            print("Niet genoeg testdata na lag van 24 uur.")
            rmse = np.nan
            rmse_results.append({
                'iteration': i + 1,
                'run_date': run_date.strftime('%Y-%m-%d'),
                'valid_predictions': 0,
                'rmse': rmse
            })
            continue

        if len(y_pred) > 0:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        else:
            rmse = np.nan

        rmse_results.append({
            'iteration': i + 1,
            'run_date': run_date.strftime('%Y-%m-%d'),
            'valid_predictions': len(y_pred),
            'rmse': rmse
        })

        print(f"Day {i+1}: ✅ {len(y_pred)} test rows, Run: {run_date.strftime('%m-%d')}")

    except Exception as e:
        print(f"Day {i+1}: ❌ Error: {e}")

# Create results dataframe
if rmse_results:
    rmse_df = pd.DataFrame(rmse_results)

    print(f"\n📊 OVERALL RMSE - Prophet Model")
    print("=" * 80)
    print(f"Successful runs: {rmse_df['rmse'].notna().sum()}/30")

    print(rmse_df[['iteration', 'run_date', 'valid_predictions', 'rmse']].round(2).to_string(index=False))

    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    print(rmse_df['rmse'].describe().round(2))

    print(f"\n📊 AVERAGE OVERALL RMSE")
    print("-" * 40)
    print(f"Mean RMSE: {rmse_df['rmse'].mean():.4f}")
    print(f"Stddev RMSE: {rmse_df['rmse'].std():.4f}")

else:
    print("❌ No runs completed successfully")

model_run_end_time = time.time()
execution_time = model_run_end_time - model_run_start_time

comments = "Base model run on 31st May with rolling df"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", rmse, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "RMSE", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv" 
# -------------------- SAVE MODEL --------------------

model_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "prophet_model.pkl"
joblib.dump(model, model_file_path)


###### Run for the BEST param

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


available_regressors = [col for col in train_data.columns if col in regressors]

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



for params in all_params:
    model = Prophet(**params)
    #for reg in available_regressors:
    for reg in FEATURES:
        model.add_regressor(reg)
    model.fit(prophet_train)
    
    forecast = model.predict(prophet_test)
    y_true = prophet_test['y'].values
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

# -------------------- Prediction on future data --------------------
"""
from datetime import timedelta

# --- Parameters ---

# Step 1: Read data
csv_file_path = PROJECT_ROOT / "src" / "data" / "warp-csv-dataset.csv"
with open(csv_file_path, 'rb') as csv_file:
    df_pd_orig = pd.read_csv(csv_file)

df = df_pd_orig.copy()


start_date = pd.Timestamp("2025-03-15 00:00:00")
rolling_days = 7  # how many days to roll
horizon = 1       # forecast 1 day at a time

all_preds, all_actuals, all_timestamps, all_horizons = [], [], [], []

# --- Daily Rolling Prediction ---
for day_offset in range(rolling_days):
    predict_date = start_date + timedelta(days=day_offset)

    # Filter forecast data for specific day
    future = df[df['target_datetime'].dt.normalize() == predict_date.normalize()].copy()

    if future.empty:
        logger.warning(f"⚠️ No data available for {predict_date.date()}, skipping.")
        continue

    # --- Prepare Input for Prophet ---
    future.rename(columns={'target_datetime': 'ds', 'Price': 'y'}, inplace=True)
    future['ds'] = pd.to_datetime(future['ds']).dt.tz_localize(None)

    # Align regressors
    if available_regressors:
        required_cols = ['ds', 'y'] + available_regressors
        future = future[required_cols]
        future[available_regressors] = future[available_regressors].ffill().bfill()
    else:
        future = future[['ds', 'y']]

    try:
        forecast = best_model.predict(future)
        y_true = future['y'].values
        y_pred = forecast['yhat'].values

        horizon_values = np.full_like(y_true, fill_value=day_offset, dtype=int)

        all_preds.extend(y_pred)
        all_actuals.extend(y_true)
        all_timestamps.extend(forecast['ds'].values)
        all_horizons.extend(horizon_values)

        logger.info(f"✅ Prediction complete for {predict_date.date()} with {len(y_pred)} rows.")

    except Exception as e:
        logger.error(f"❌ Prediction failed for {predict_date.date()}: {e}")


# === Final Evaluation ===
if not all_preds:
    logger.warning("❌ No predictions were made. Check your data coverage.")
else:
    # Overall Metrics
    final_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    final_mae = mean_absolute_error(all_actuals, all_preds)
    final_r2 = r2_score(all_actuals, all_preds)

    logger.info(f"🎯 Rolling RMSE: {final_rmse:.3f} | MAE: {final_mae:.3f} | R²: {final_r2:.3f}")

    # --- Combine to DataFrame ---
    df_eval = pd.DataFrame({
        'Timestamp': all_timestamps,
        'Actual': all_actuals,
        'Predicted': all_preds,
        'Horizon': all_horizons
    })

    logger.info(f"✅ Evaluation DataFrame created with {len(df_eval)} rows")

    # --- Pivot for per-horizon prediction view ---
    pivot_df = df_eval.pivot_table(index='Timestamp', columns='Horizon', values='Predicted', aggfunc='first')
    pivot_df.columns = [f'Predicted_{int(h+1)}d_ahead' for h in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Add Actuals
    actuals = df_eval.drop_duplicates('Timestamp')[['Timestamp', 'Actual']]
    pivot_df = pd.merge(pivot_df, actuals, on='Timestamp', how='left')

    print(pivot_df.head())

    # === Save full prediction table ===
    forecast_output_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "rolling_predictions.csv"
    pivot_df.to_csv(forecast_output_path, index=False)
    logger.info(f"📁 Rolling predictions saved to: {forecast_output_path}")

    # === Horizon-wise RMSE ===
    horizon_rmse_log = {}
    logger.info("📉 Horizon-wise RMSEs:")
    for h in sorted(df_eval['Horizon'].unique()):
        df_h = df_eval[df_eval['Horizon'] == h]
        if not df_h.empty:
            rmse_h = np.sqrt(mean_squared_error(df_h['Actual'], df_h['Predicted']))
            logger.info(f"🔹 Horizon {h+1}-step ahead → RMSE: {rmse_h:.4f}")
            horizon_rmse_log[f"rmse_horizon_{h+1}"] = rmse_h

    # Save RMSEs
    horizon_rmse_df = pd.DataFrame(list(horizon_rmse_log.items()), columns=["Horizon", "RMSE"])
    horizon_model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "horizon_wise_rmse.csv"
    horizon_rmse_df.to_csv(horizon_model_results_file_path, index=False)
    logger.info("✅ Horizon-wise RMSEs saved.")
"""
