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

print("=" * 60)

# Rolling forecast loop
for day_offset in range(rolling_days):
    predict_date = start_date + timedelta(days=day_offset)
    
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

    
        # Predict
        forecast = model.predict(future)

        if forecast.empty:
            logger.error(f"❌ Forecast returned empty DataFrame for {predict_date.date()}")
            continue

        y_true = future['y'].values
        y_pred = forecast['yhat'].values

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



