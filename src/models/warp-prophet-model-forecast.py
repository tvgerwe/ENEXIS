import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import timedelta


from pathlib import Path
import logging
import json
import sqlite3

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(Path(__file__).parent / "logs" / "warp-prophet_forecast_tuned_model-json.log"),
    filemode='a'
)
logger = logging.getLogger('prophet_forecast_tuned_model')

# === Config Setup ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

model_file_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "prophet_hyper_tuned_model.pkl"
forecast_output_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "forecast_vs_actual.csv"
forecast_output_path_rolling = PROJECT_ROOT / "src" / "models" / "model_run_results" / "forecast_vs_actual_rolling.csv"
model_metrics_results_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "model_run_metrics.csv"
model_metrics_results_path_rolling = PROJECT_ROOT / "src" / "models" / "model_run_results" / "model_run_metrics-rolling.csv"

# === Load model ===
model = joblib.load(model_file_path)
logger.info("✅ Prophet model loaded from disk.")


# === Paths and Config ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
csv_file_path = PROJECT_ROOT / "src" / "data" / "warp-csv-dataset.csv"

# === Load Data ===
df = pd.read_csv(csv_file_path)
# print(df.head)

df['target_datetime'] = pd.to_datetime(df['target_datetime'], errors='coerce')

# === Load Model ===
model = joblib.load(model_file_path)

# === Parameters ===
start_date = pd.Timestamp("2025-03-15 00:00:00")
rolling_days = 7  # number of days to roll
horizon = 1       # forecast horizon per roll

regressors = [
    'Load', 'shortwave_radiation', 'temperature_2m', 'direct_normal_irradiance', 'diffuse_radiation',
    'Flow_NO', 'yearday_cos', 'Flow_GB', 'month', 'is_dst', 'yearday_sin', 'is_non_working_day',
    'hour_cos', 'is_weekend', 'cloud_cover', 'weekday_sin', 'hour_sin', 'weekday_cos'
]
available_regressors = [col for col in df.columns if col in regressors]

# === Storage ===
all_preds, all_actuals, all_timestamps, all_horizons = [], [], [], []
#print(f"Available regressors: {available_regressors}")
from datetime import timedelta
import numpy as np
import pandas as pd

# Ensure datetime column is correct
df['target_datetime'] = pd.to_datetime(df['target_datetime'], errors='coerce').dt.tz_localize(None)
df['target_datetime'] = df['target_datetime'].dt.normalize()

print(f"🔁 Starting rolling forecast from: {start_date.date()} for {rolling_days} days with horizon {horizon} day(s)")
print("=" * 60)

# Rolling forecast loop
for day_offset in range(rolling_days):
    predict_date = start_date + timedelta(days=day_offset)
    print(f"\n📅 Forecasting for day offset {day_offset} → {predict_date.date()}")

    # Filter target day
    target_day = df[df['target_datetime'] == predict_date.normalize()].copy()

    if target_day.empty:
        logger.warning(f"⚠️ No data found for {predict_date.date()}, skipping.")
        continue

    try:
        # Rename for Prophet format
        target_day.rename(columns={'target_datetime': 'ds', 'Price': 'y'}, inplace=True)
        target_day['ds'] = pd.to_datetime(target_day['ds']).dt.tz_localize(None)

        # Select required columns
        if available_regressors:
            cols_needed = ['ds', 'y'] + available_regressors
            future = target_day[cols_needed].copy()
            future[available_regressors] = future[available_regressors].ffill().bfill()
        else:
            future = target_day[['ds', 'y']].copy()

        print(f"🧾 Future data shape: {future.shape}")
        print(future.head())

        # Predict
        forecast = model.predict(future)

        if forecast.empty:
            logger.error(f"❌ Forecast returned empty DataFrame for {predict_date.date()}")
            continue

        y_true = future['y'].values
        y_pred = forecast['yhat'].values

        print(f"✅ y_true ({len(y_true)}): {y_true[:5]}")
        print(f"✅ y_pred ({len(y_pred)}): {y_pred[:5]}")

        if len(y_pred) == 0:
            logger.warning(f"⚠️ No y_pred returned for {predict_date.date()}, skipping.")
            continue

        horizons = np.full_like(y_true, fill_value=day_offset, dtype=int)

        all_preds.extend(y_pred)
        all_actuals.extend(y_true)
        all_timestamps.extend(forecast['ds'].values)
        all_horizons.extend(horizons)

        logger.info(f"✅ Prediction complete for {predict_date.date()} with {len(y_pred)} rows.")

    except Exception as e:
        logger.exception(f"❌ Error while predicting for {predict_date.date()}: {e}")

# === Evaluation ===
if not all_preds:
    logger.warning("❌ No predictions were made. Check your data coverage.")
else:
    final_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    final_mae = mean_absolute_error(all_actuals, all_preds)
    final_r2 = r2_score(all_actuals, all_preds)

    logger.info(f"🎯 Rolling RMSE: {final_rmse:.3f} | MAE: {final_mae:.3f} | R²: {final_r2:.3f}")

    # Evaluation DataFrame
    df_eval = pd.DataFrame({
        'Timestamp': all_timestamps,
        'Actual': all_actuals,
        'Predicted': all_preds,
        'Horizon': all_horizons
    })

    # Pivot for multi-horizon view
    pivot_df = df_eval.pivot_table(index='Timestamp', columns='Horizon', values='Predicted', aggfunc='first')
    pivot_df.columns = [f'Predicted_{h+1}d_ahead' for h in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    actuals = df_eval.drop_duplicates('Timestamp')[['Timestamp', 'Actual']]
    pivot_df = pd.merge(pivot_df, actuals, on='Timestamp', how='left')

    # Save predictions
    forecast_output_path = PROJECT_ROOT / "src" / "models" / "model_run_results" / "rolling_predictions.csv"
    pivot_df.to_csv(forecast_output_path, index=False)
    logger.info(f"📁 Rolling predictions saved to: {forecast_output_path}")

    # Horizon-wise RMSE
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
# Step 1: Read data
prediction_csv_file_path = PROJECT_ROOT / "src" / "data" / "warp-csv-prediction-dataset.csv"
with open(prediction_csv_file_path, 'rb') as csv_file:
    df_pd_prediction_orig = pd.read_csv(prediction_csv_file_path)

# Ensure 'target_datetime' is parsed as timezone-naive datetime
df_pd_prediction_orig['target_datetime'] = pd.to_datetime(df_pd_prediction_orig['target_datetime'], errors='coerce')
df_pd_prediction_orig['target_datetime'] = df_pd_prediction_orig['target_datetime'].dt.tz_localize(None)

print("Loaded data sample:")
print(df_pd_prediction_orig[['target_datetime', 'Price']].head(20))
# Get all rows with nonzero Price values, sorted by Price and date
all_nonzero_prices = df_pd_prediction_orig[df_pd_prediction_orig['Price'] != 0.0][['target_datetime', 'Price']].sort_values(['Price', 'target_datetime'])

print("All dates for each unique nonzero Price value:")
print(all_nonzero_prices)

print("Unique Price values:", df_pd_prediction_orig['Price'].unique())

# Print dates where Price is not 0.0 and date is after 25 May 2025
nonzero_price_rows = df_pd_prediction_orig[
    (df_pd_prediction_orig['Price'] != 0.0) & 
    (df_pd_prediction_orig['target_datetime'] > pd.to_datetime('2025-05-27'))
]
print("Nonzero prices after 2025-05-27:")
print(nonzero_price_rows[['target_datetime', 'Price']])


# === Prepare DataFrame for Prophet ===
df_pd_orig['datetime'] = df_pd_orig['target_datetime']
df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])

df = df_pd_orig.sort_values(by='datetime')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
df['ds'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
df['y'] = df['Price']

#regressors = [
#    'Load','shortwave_radiation','temperature_2m','direct_normal_irradiance','diffuse_radiation','Flow_NO','yearday_cos','Flow_GB','month',
#    'is_dst','yearday_sin','is_non_working_day','hour_cos','is_weekend','cloud_cover','weekday_sin','hour_sin','weekday_cos',
#]

FEATURES = [
    'Load', 'shortwave_radiation', 'temperature_2m', 
    'Flow_NO', 'yearday_cos', 'Flow_GB', 
    'yearday_sin', 'wind_speed_10m', 
    'hour_sin']

regressors = FEATURES

#regressors = [
#    'Load','shortwave_radiation','temperature_2m','direct_normal_irradiance','diffuse_radiation',
#    'Flow_NO','yearday_cos','Flow_GB','month','is_dst','yearday_sin','is_non_working_day',
#    'hour_cos','is_weekend','cloud_cover','weekday_sin','hour_sin','weekday_cos'
#]

# wind_speed_10m  removed due as it was not in model build

logger.info(f"✅ Data loaded and prepared. Total records: {len(df)}")

# === 1. Normal Forecast (single window) ===
# User-defined start date for the forecast window
user_forecast_start = pd.to_datetime("2025-05-15")  # <-- Set your desired start date here
forecast_horizon_days = 6
forecast_start = user_forecast_start
forecast_end = forecast_start + pd.Timedelta(days=forecast_horizon_days - 1)


future = df[(df['ds'] >= forecast_start) & (df['ds'] <= forecast_end)].copy()
future_pred = future[['ds'] + regressors].copy()
if future_pred[regressors].isnull().any().any():
    logger.warning("Missing values found in regressors for normal forecast, applying ffill/bfill.")
    future_pred[regressors] = future_pred[regressors].ffill().bfill()

forecast = model.predict(future_pred)
yhat = forecast[['ds', 'yhat']]
actual = future[['ds', 'y']]
merged = actual.merge(yhat, on='ds', how='left')

mae = mean_absolute_error(merged['y'], merged['yhat'])
mse = mean_squared_error(merged['y'], merged['yhat'])
rmse = np.sqrt(mse)
r2 = r2_score(merged['y'], merged['yhat'])

logger.info(f"Normal Forecast metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# Save normal forecast results
df_forecast = merged.rename(columns={'y': 'actual', 'yhat': 'predicted'})
df_forecast.to_csv(forecast_output_path, index=False)
logger.info(f"📁 Normal Forecast vs Actual saved to: {forecast_output_path}")

df_model_metrics = pd.DataFrame([{
    "run": 1,
    "start": forecast_start,
    "end": forecast_end,
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}])
df_model_metrics.to_csv(model_metrics_results_path, index=False)
logger.info(f"Normal forecast metrics saved to: {model_metrics_results_path}")

# === 2. Rolling Window Forecast ===
start_date = pd.to_datetime("2025-05-15") + timedelta(hours=36)
num_rolling_runs = 6
horizon = 6  # days ahead

'''
# Print dates where Price is not 0.0 and date is after 25 May 2025
nonzero_price_rows = df[(df['Price'] != 0.0) & (df['ds'] > pd.to_datetime('2025-05-25'))]
print(nonzero_price_rows[['ds', 'Price']])
'''

forecast_rows = []
model_run_metrics = []

for i in range(num_rolling_runs):
    predict_start = start_date + pd.Timedelta(days=i)
    predict_end = predict_start + pd.Timedelta(days=horizon - 1)
    logger.info(f"Rolling window {i+1}: {predict_start} to {predict_end}")
    print(f"Rolling window {i+1}: {predict_start} to {predict_end}")

    future = df[(df['ds'] >= predict_start) & (df['ds'] <= predict_end)].copy()
    missing_regs = [r for r in regressors if r not in future.columns]
    if missing_regs:
        logger.error(f"Missing regressors in prediction window: {missing_regs}")
        raise ValueError(f"Missing regressors in prediction window: {missing_regs}")

    future_pred = future[['ds'] + regressors].copy()
    if future_pred[regressors].isnull().any().any():
        logger.warning(f"Missing values found in regressors for window {i+1}, applying ffill/bfill.")
        future_pred[regressors] = future_pred[regressors].ffill().bfill()

    forecast = model.predict(future_pred)
    yhat = forecast[['ds', 'yhat']]
    actual = future[['ds', 'y']]
    merged = actual.merge(yhat, on='ds', how='left')

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    mse = mean_squared_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(merged['y'], merged['yhat'])

    logger.info(f"Window {i+1} metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    for _, row in merged.iterrows():
        forecast_rows.append({
            "ds": row['ds'],
            "actual": row['y'],
            "predicted": row['yhat'],
            "window_start": predict_start,
            "window_end": predict_end,
            "rmse": rmse
        })

    model_run_metrics.append({
        "run": i + 1,
        "start": predict_start,
        "end": predict_end,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    })
    '''
    print(future[['ds', 'Price']])
    print(f"Nonzero prices: {future['Price'][future['Price'] != 0]}")
    '''

# Save rolling window forecast results
try:
    df_forecast_rolling = pd.DataFrame(forecast_rows)
    # Round 'actual' to 3 decimals, preserving negative values
    if 'actual' in df_forecast_rolling.columns:
        df_forecast_rolling['actual'] = df_forecast_rolling['actual'].round(3)
    df_forecast_rolling.to_csv(forecast_output_path_rolling, index=False)
    logger.info(f"📁 Rolling Window Forecast vs Actual saved to: {forecast_output_path_rolling}")
except Exception as e:
    logger.error(f"Failed to save rolling window forecast vs actual: {e}")
    raise

try:
    df_model_metrics_rolling = pd.DataFrame(model_run_metrics)
    df_model_metrics_rolling.to_csv(model_metrics_results_path_rolling, index=False)
    logger.info(f"Rolling window model run metrics saved to: {model_metrics_results_path_rolling}")
except Exception as e:
    logger.error(f"Failed to save rolling window model run metrics: {e}")
    raise

"""

