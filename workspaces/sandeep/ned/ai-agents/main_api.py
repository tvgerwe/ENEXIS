# main_api.py
# SUMMARY: Unified FastAPI server for the modular Prophet pipeline. Exposes endpoints for model build, forecast, and cross-validation, delegating to agent_orchestration.py for business logic. Supports file upload, robust logging, and error handling. Entry point for API-driven workflows and integration with external systems.

import logging
from fastapi import FastAPI, File, UploadFile, Form
import os
from agent_orchestration import build_and_save_prophet_model, forecast_with_saved_model, run_agent_pipeline_forecast_only

app = FastAPI()

# Configure logger
logger = logging.getLogger("main_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Core ML functions (replace with your actual logic) ---
def build_model_core(csv_path, train_start, train_end, test_start, test_end, regressors):
    logger.info(f"[build_model_core] Delegating to build_and_save_prophet_model with {csv_path}, train: {train_start}-{train_end}, test: {test_start}-{test_end}, regressors: {regressors}")
    # Call the existing build_and_save_prophet_model function with positional arguments
    return build_and_save_prophet_model(
        csv_path,
        train_start,
        train_end,
        test_start,
        test_end,
        regressors
    )

def forecast_core(csv_path, regressors, periods, freq):
    logger.info(f"[forecast_core] Delegating to forecast_with_saved_model with {csv_path}, regressors: {regressors}, periods: {periods}, freq: {freq}")
    return forecast_with_saved_model(
        csv_path,
        regressors,
        periods,
        freq
    )

def cross_validate_core(csv_path, regressors, initial, period, horizon):
    logger.info(f"[cross_validate_core] Delegating to run_agent_pipeline_forecast_only with {csv_path}, regressors: {regressors}, initial: {initial}, period: {period}, horizon: {horizon}")
    return run_agent_pipeline_forecast_only(
        csv_path,
        regressors,
        initial,
        period,
        horizon
    )

# --- API Endpoints ---
@app.post("/build-model")
def build_model(
    file: UploadFile = File(...),
    train_start: str = Form(...),
    train_end: str = Form(...),
    test_start: str = Form(...),
    test_end: str = Form(...),
    regressors: str = Form(...)
):
    logger.info("Received /build-model request")
    temp_path = "temp_build_model.csv"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    logger.info(f"Saved uploaded file to {temp_path}")
    result = build_model_core(
        csv_path=temp_path,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        regressors=regressors
    )
    os.remove(temp_path)
    logger.info(f"Removed temp file {temp_path}")
    if result.get("success"):
        logger.info(f"Build model call successful: {result}")
    else:
        logger.error(f"Build model call failed: {result}")
    logger.info(f"Returning result: {result}")
    return result

@app.post("/forecast")
def forecast(
    file: UploadFile = File(...),
    regressors: str = Form(...),
    periods: int = Form(...),
    freq: str = Form("D")
):
    logger.info("Received /forecast request")
    temp_path = "temp_forecast.csv"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    logger.info(f"Saved uploaded file to {temp_path}")
    result = forecast_core(
        csv_path=temp_path,
        regressors=regressors,
        periods=periods,
        freq=freq
    )
    os.remove(temp_path)
    logger.info(f"Removed temp file {temp_path}")
    if result.get("success"):
        logger.info(f"Forecast call successful: {result}")
    else:
        logger.error(f"Forecast call failed: {result}")
    logger.info(f"Returning result: {result}")
    return result

@app.post("/cross-validate")
def cross_validate(
    file: UploadFile = File(...),
    regressors: str = Form(...),
    initial: str = Form(...),
    period: str = Form(...),
    horizon: str = Form(...)
):
    logger.info("Received /cross-validate request")
    temp_path = "temp_validate.csv"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    logger.info(f"Saved uploaded file to {temp_path}")
    result = cross_validate_core(
        csv_path=temp_path,
        regressors=regressors,
        initial=initial,
        period=period,
        horizon=horizon
    )
    os.remove(temp_path)
    logger.info(f"Removed temp file {temp_path}")
    if result.get("success"):
        logger.info(f"Cross-validate call successful: {result}")
    else:
        logger.error(f"Cross-validate call failed: {result}")
    logger.info(f"Returning result: {result}")
    return result