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
    'Load', 'shortwave_radiation', 'temperature_2m', 'direct_normal_irradiance', 'diffuse_radiation',
    'Flow_NO', 'yearday_cos', 'Flow_GB', 'month', 'is_dst', 'yearday_sin', 'is_non_working_day',
    'hour_cos', 'is_weekend', 'cloud_cover', 'weekday_sin', 'hour_sin', 'weekday_cos'
]
target = 'Price'

# Initial training window
base_start = "2023-01-01 00:00:00"
base_end = "2024-12-31 23:00:00"
base_run = "2025-01-01 00:00:00"

rmse_results = []
results = []
experiment_runs = []  # Array to store experiment results

###### Base Model Run

print("🔍 Testing Prophet Model as base run")
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
        train_data = train_data.dropna(subset=['target_datetime', target])
        test_data = test_data.dropna(subset=['target_datetime', target])

        if test_data.empty or train_data.empty:
            print(f"Day {i+1}: ❌ Not enough data for training or testing")
            continue

        # Prepare data for Prophet
        prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y']]
        prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
        prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y']]
        prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

        # Train Prophet model with extra regressors
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_train)

        # Forecast for the test period
        future = prophet_test[['ds']]
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
    
    # Add experiment result
    experiment_runs.append({
        'Iteration': 'First',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'None',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

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

comments = "Base model run on 31st May with 10 features"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", rmse, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "RMSE", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv" 

######## Top 6 Features

print("🔍 Testing Prophet Model Top 6 features - RMSE per forecast day")
print("=" * 60)

TOP_6_FEATURES = [
    'Load','shortwave_radiation','temperature_2m','direct_normal_irradiance','diffuse_radiation','Flow_NO'
]

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
        train_data = train_data.dropna(subset=['target_datetime', target] + TOP_6_FEATURES)
        test_data = test_data.dropna(subset=['target_datetime', target] + TOP_6_FEATURES)

        if test_data.empty or train_data.empty:
            print(f"Day {i+1}: ❌ Not enough data for training or testing")
            continue

        # Prepare data for Prophet
        prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TOP_6_FEATURES]
        prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
        prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TOP_6_FEATURES]
        prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

        # Train Prophet model with extra regressors
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        for reg in TOP_6_FEATURES:
            model.add_regressor(reg)
        model.fit(prophet_train)

        # Forecast for the test period
        future = prophet_test[['ds'] + TOP_6_FEATURES]
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
    
    # Add experiment result
    experiment_runs.append({
        'Iteration': 'Fifth',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'Top 6 Corr Features',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

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

####### Time Features

print("🔍 Testing Prophet Model Time features - RMSE per forecast day")
print("=" * 60)

TIME_FEATURES = [
    'Load', 'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin', 'yearday_cos', 'yearday_sin'
]

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
        train_data = train_data.dropna(subset=['target_datetime', target] + TIME_FEATURES)
        test_data = test_data.dropna(subset=['target_datetime', target] + TIME_FEATURES)

        if test_data.empty or train_data.empty:
            print(f"Day {i+1}: ❌ Not enough data for training or testing")
            continue

        # Prepare data for Prophet
        prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TIME_FEATURES]
        prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
        prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TIME_FEATURES]
        prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

        # Train Prophet model with extra regressors
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        for reg in TIME_FEATURES:
            model.add_regressor(reg)
        model.fit(prophet_train)

        # Forecast for the test period
        future = prophet_test[['ds'] + TIME_FEATURES]
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
    
    # Add experiment result
    experiment_runs.append({
        'Iteration': 'Second',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'Time Columns',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

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

comments = "Base model run on 31st May with time features"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", rmse, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "RMSE", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv" 

####### All features

print("🔍 Testing Prophet Model with all selected features")
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
    
    # Add experiment result
    experiment_runs.append({
        'Iteration': 'Third',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'All Corr Features',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

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

######## Top 10 Features

print("🔍 Testing Prophet Model Top 10 features - RMSE per forecast day")
print("=" * 60)

TOP_10_FEATURES = [
    'Load', 'shortwave_radiation', 'temperature_2m', 'direct_normal_irradiance',
    'diffuse_radiation', 'Flow_NO', 'yearday_cos', 'Flow_GB', 'month', 'is_dst'
]

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
        train_data = train_data.dropna(subset=['target_datetime', target] + TOP_10_FEATURES)
        test_data = test_data.dropna(subset=['target_datetime', target] + TOP_10_FEATURES)

        if test_data.empty or train_data.empty:
            print(f"Day {i+1}: ❌ Not enough data for training or testing")
            continue

        # Prepare data for Prophet
        prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TOP_10_FEATURES]
        prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
        prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TOP_10_FEATURES]
        prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

        # Train Prophet model with extra regressors
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        for reg in TOP_10_FEATURES:
            model.add_regressor(reg)
        model.fit(prophet_train)

        # Forecast for the test period
        future = prophet_test[['ds'] + TOP_10_FEATURES]
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
    
    # Add experiment result
    experiment_runs.append({
        'Iteration': 'Fourth',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'Top 10 Corr Features',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

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

comments = "Base model run on 31st May with 10 features"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", rmse, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "RMSE", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv" 

######## Top 4 Features

print("🔍 Testing Prophet Model Top 4 features - RMSE per forecast day")
print("=" * 60)

TOP_4_FEATURES = [
    'Load', 'shortwave_radiation', 'temperature_2m', 'direct_normal_irradiance'
]

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
        train_data = train_data.dropna(subset=['target_datetime', target] + TOP_4_FEATURES)
        test_data = test_data.dropna(subset=['target_datetime', target] + TOP_4_FEATURES)

        if test_data.empty or train_data.empty:
            print(f"Day {i+1}: ❌ Not enough data for training or testing")
            continue

        # Prepare data for Prophet
        prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TOP_4_FEATURES]
        prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
        prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + TOP_4_FEATURES]
        prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

        # Train Prophet model with extra regressors
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        for reg in TOP_4_FEATURES:
            model.add_regressor(reg)
        model.fit(prophet_train)

        # Forecast for the test period
        future = prophet_test[['ds'] + TOP_4_FEATURES]
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
    
    # Add experiment result
    experiment_runs.append({
        'Iteration': 'Sixth',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'Top 4 Corr Features',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

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

# Save experiment results
experiment_df = pd.DataFrame(experiment_runs)
experiment_results_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "experiment_results.csv"
experiment_df.to_csv(experiment_results_path, index=False)

# Print the experiment results in a formatted table
print("\n📊 EXPERIMENT RESULTS")
print("=" * 80)
print(experiment_df.to_string(index=False))

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

# Prepare data for Prophet
prophet_train = train_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + FEATURES]
prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
prophet_test = test_data.rename(columns={'target_datetime': 'ds', target: 'y'})[['ds', 'y'] + FEATURES]
prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

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

####### All features with best params

print("🔍 Testing Prophet Model with all selected features and best params")
print("=" * 60)

#2025-05-31 16:10:13,834 - warp-prophet-model - INFO - ✅ Best Parameters: {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 
# 'seasonality_prior_scale': 5.0, 'holidays_prior_scale': 1.0, 'changepoint_range': 0.8, 'n_changepoints': 50}

best_param_grid = {
    'seasonality_mode': ['additive'],
    'changepoint_prior_scale': [0.01],
    'seasonality_prior_scale': [5.0],
    'holidays_prior_scale': [1.0],
    'changepoint_range': [0.8],
    'n_changepoints': [50],
    'daily_seasonality': [True],
    'weekly_seasonality': [True],
    'yearly_seasonality': [True]
}

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
        """
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            changepoint_range=params['changepoint_range'],
            n_changepoints=params['n_changepoints'],
            daily_seasonality=params.get('daily_seasonality', True),
            weekly_seasonality=params.get('weekly_seasonality', True),
            yearly_seasonality=params.get('yearly_seasonality', True)
        )

        for reg in FEATURES:
            model.add_regressor(reg)
        model.fit(prophet_train)
        """

        forecast = best_model.predict(prophet_test)

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

    experiment_runs.append({
        'Iteration': 'Seventh',
        'Model Configs': 'Prophet',
        'Feature and Regressors': 'Best Param',
        'RMSE': f"{rmse_df['rmse'].mean():.4f}",
        'StdDev RMSE': f"{rmse_df['rmse'].std():.4f}"
    })

else:
    print("❌ No runs completed successfully")

model_run_end_time = time.time()
execution_time = model_run_end_time - model_run_start_time

comments = "Base model run on 1 Jun with with 2 year of tran data and 3 month of test data"
model_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results.append(["Prophet", rmse, comments, execution_time, model_run_timestamp])
metrics_df = pd.DataFrame(results, columns=["Model", "RMSE", "Comments", "Execution Time", "Run At"])

model_results_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "warp-prophet-model-results.csv" 

# Save experiment results
experiment_df = pd.DataFrame(experiment_runs)
experiment_results_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "experiment_results.csv"
experiment_df.to_csv(experiment_results_path, index=False)

# Print the experiment results in a formatted table
print("\n📊 EXPERIMENT RESULTS")
print("=" * 80)
print(experiment_df.to_string(index=False))



