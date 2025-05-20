# agent_api_calls.py
# SUMMARY: Python client library for programmatically calling the FastAPI endpoints in main_api.py. Provides robust, logged functions for model build, forecast, and cross-validation. Used by orchestration scripts and CLI/scheduler for API-driven workflows.

# Purpose: Python client functions for calling the unified FastAPI endpoints in main_api.py
import requests
import logging

# Configure logger
logger = logging.getLogger("agent_api_calls")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def call_build_model_api(
    csv_path,
    train_start,
    train_end,
    test_start,
    test_end,
    regressors,
    api_url="http://localhost:9000/build-model"
):
    logger.info(f"Calling unified /build-model API at {api_url} with file {csv_path}")
    with open(csv_path, "rb") as f:
        files = {"file": ("data.csv", f, "text/csv")}
        data = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "regressors": regressors
        }
        response = requests.post(api_url, files=files, data=data)
    try:
        logger.info(f"/build-model API response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error parsing /build-model API response: {e}")
        return {"success": False, "error_code": type(e).__name__, "error_message": str(e)}

def call_forecast_api(csv_path, regressors, periods, freq="D", api_url="http://localhost:9000/forecast"):
    logger.info(f"Calling unified /forecast API at {api_url} with file {csv_path}, periods={periods}, freq={freq}")
    with open(csv_path, "rb") as f:
        files = {"file": ("data.csv", f, "text/csv")}
        data = {
            "regressors": regressors,
            "periods": periods,
            "freq": freq
        }
        response = requests.post(api_url, files=files, data=data)
    try:
        logger.info(f"/forecast API response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error parsing /forecast API response: {e}")
        return {"success": False, "error_code": type(e).__name__, "error_message": str(e)}

def call_cross_validate_api(csv_path, regressors, initial, period, horizon, api_url="http://localhost:9000/cross-validate"):
    logger.info(f"Calling unified /cross-validate API at {api_url} with file {csv_path}, initial={initial}, period={period}, horizon={horizon}")
    with open(csv_path, "rb") as f:
        files = {"file": ("data.csv", f, "text/csv")}
        data = {
            "regressors": regressors,
            "initial": initial,
            "period": period,
            "horizon": horizon
        }
        response = requests.post(api_url, files=files, data=data)
    try:
        logger.info(f"/cross-validate API response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error parsing /cross-validate API response: {e}")
        return {"success": False, "error_code": type(e).__name__, "error_message": str(e)}
