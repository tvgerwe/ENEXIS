# main.py
# SUMMARY: Scheduler and CLI entry point for orchestrating the modular Prophet pipeline. Uses agent_api_calls.py to call the unified API endpoints for model build, forecast, and cross-validation. Supports scheduled and manual runs for automated time series workflows.

# Purpose: Scheduler/CLI entry point for running agent orchestration pipelines.
import schedule
import time
from agent_api_calls import (
    call_build_model_api,
    call_forecast_api,
    call_cross_validate_api
)

# Example constants (customize as needed)
TARGET_TIME_BUILD = "14:36"
TARGET_TIME_FORECAST_ONLY = "14:10"
DATA_FILE_PATH = "src/data/warp-csv-dataset.csv"
REGRESSORS = "month,shortwave_radiation,apparent_temperature,temperature_2m,direct_normal_irradiance,diffuse_radiation,yearday_sin,Flow_BE,hour_sin,is_non_working_day,is_weekend,is_holiday,weekday_cos,wind_speed_10m,hour_cos,weekday_sin,cloud_cover,Flow_GB,Nuclear_Vol,yearday_cos,Flow_NO,Load"
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-03-14"
TEST_START = "2025-03-15"
TEST_END = "2025-04-14"

# Scheduler jobs

def run_build_and_forecast():
    print("[Scheduler] Running build and forecast pipeline...")
    build_result = call_build_model_api(
        csv_path=DATA_FILE_PATH,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        regressors=REGRESSORS
    )
    print("Build result:", build_result)
    forecast_result = call_forecast_api(
        csv_path=DATA_FILE_PATH,
        regressors=REGRESSORS,
        periods=30
    )
    print("Forecast result:", forecast_result)

def run_forecast_only():
    print("[Scheduler] Running forecast-only pipeline...")
    forecast_result = call_forecast_api(
        csv_path=DATA_FILE_PATH,
        regressors=REGRESSORS,
        periods=30
    )
    print("Forecast result:", forecast_result)

schedule.every().day.at(TARGET_TIME_BUILD).do(run_build_and_forecast)
schedule.every().day.at(TARGET_TIME_FORECAST_ONLY).do(run_forecast_only)

print(f"Scheduler started. The agent pipeline will run model build+forecast at {TARGET_TIME_BUILD} and forecast-only at {TARGET_TIME_FORECAST_ONLY} (Eindhoven time).")
while True:
    schedule.run_pending()
    time.sleep(1)