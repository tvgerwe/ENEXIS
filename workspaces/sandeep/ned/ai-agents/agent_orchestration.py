# agent_orchestration.py
# SUMMARY: Orchestration and business logic layer for the modular Prophet pipeline. Provides functions for model build, forecast, and cross-validation by calling the core logic directly. Used by main_api.py (API server) and main.py (scheduler/CLI) for unified, non-HTTP orchestration.
# Purpose: Orchestration/business logic for model build, forecast, and cross-validation.
# This module should call the core logic directly, not via HTTP. If you want to orchestrate via API, import agent_api_calls.py.
import logging
from prophet_api import build_and_save_prophet_model_core
from prophet_api_forecast import forecast_with_saved_model_core
from prophet_crossvalidation_core import cross_validate_model_core

# Configure logger
logger = logging.getLogger("agent_orchestration")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Example constants (customize as needed)
TARGET_TIME_BUILD = "14:36"
TARGET_TIME_FORECAST_ONLY = "14:10"
ACCEPTABLE_RMSE_THRESHOLD = 0.5
DATA_FILE_PATH = "src/data/warp-csv-dataset.csv"
TARGET_COLUMN = "Price"
REGRESSORS = "month,shortwave_radiation,apparent_temperature,temperature_2m,direct_normal_irradiance,diffuse_radiation,yearday_sin,Flow_BE,hour_sin,is_non_working_day,is_weekend,is_holiday,weekday_cos,wind_speed_10m,hour_cos,weekday_sin,cloud_cover,Flow_GB,Nuclear_Vol,yearday_cos,Flow_NO,Load"
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-03-14"
TEST_START = "2025-03-15"
TEST_END = "2025-04-14"

# Import your core model logic here (not HTTP client functions)
# from prophet_core import build_and_save_prophet_model, forecast_with_saved_model, cross_validate_model

def build_and_save_prophet_model(csv_path, train_start, train_end, test_start, test_end, regressors):
    logger.info("--- Running Prophet Model Training (core logic) ---")
    result = build_and_save_prophet_model_core(csv_path, train_start, train_end, test_start, test_end, regressors)
    logger.info(f"Prophet Model Result: {result}")
    return result

def forecast_with_saved_model(csv_path, regressors, periods, freq="D"):
    logger.info("--- Running Prophet Forecast (core logic) ---")
    result = forecast_with_saved_model_core(csv_path, regressors, periods, freq)
    logger.info(f"Forecast Result: {result}")
    return result

def cross_validate_model(csv_path, regressors, initial, period, horizon):
    logger.info("--- Running Prophet Cross-Validation (core logic) ---")
    result = cross_validate_model_core(csv_path, regressors, initial, period, horizon)
    logger.info(f"Cross-Validation Result: {result}")
    return result

# Example orchestration pipeline (calls core logic)
def run_agent_pipeline_build_and_forecast():
    build_and_save_prophet_model(
        csv_path=DATA_FILE_PATH,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        regressors=REGRESSORS
    )
    forecast_with_saved_model(
        csv_path=DATA_FILE_PATH,
        regressors=REGRESSORS,
        periods=30
    )

def run_agent_pipeline_forecast_only(csv_path, regressors, initial, period, horizon):
    return cross_validate_model(
        csv_path=csv_path,
        regressors=regressors,
        initial=initial,
        period=period,
        horizon=horizon
    )