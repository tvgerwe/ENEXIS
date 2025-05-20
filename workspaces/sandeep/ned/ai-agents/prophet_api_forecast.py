# prophet_api_forecast.py
# SUMMARY: FastAPI microservice and core logic for Prophet-based forecasting using saved models. Provides a core function for generating forecasts from trained Prophet models, handling file uploads, regressor management, and robust time range logic. Used by the unified API and orchestration layer for modular time series forecasting workflows.

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from prophet import Prophet
import io
import joblib
import os
from datetime import datetime, timedelta

app = FastAPI()

MODEL_PATH = "/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/workspaces/sandeep/ned/ai-agents/prophet-ai-agent-model.pkl"
REGRESSORS_PATH = "/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/workspaces/sandeep/ned/prophet_ai-agent-model-regressors.txt"
CSV_FILE_PATH = "/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/warp-csv-dataset.csv"

def forecast_with_saved_model_core(
    csv_path,
    regressors,
    periods,
    freq="H",
    start_datetime=None,
    forecast_days=6
):
    try:
        if not os.path.exists(MODEL_PATH):
            return {
                "success": False,
                "error_code": "ModelNotFound",
                "error_message": f"Trained model not found at {MODEL_PATH}"
            }
        model = joblib.load(MODEL_PATH)
        if os.path.exists(REGRESSORS_PATH):
            with open(REGRESSORS_PATH, "r") as f:
                trained_regressors = [line.strip() for line in f if line.strip()]
        else:
            trained_regressors = [col.strip() for col in regressors.split(",") if col.strip()]
        # Robustly handle both file paths and file-like objects
        if isinstance(csv_path, (str, bytes, os.PathLike)):
            df = pd.read_csv(csv_path)
        else:
            csv_path.seek(0)
            df = pd.read_csv(csv_path)
        print("Loaded DataFrame shape:", df.shape)
        print("Loaded DataFrame head:\n", df.head())
        # Always rename 'datetime' to 'ds' if present
        if 'ds' not in df.columns and 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'ds'})
        print("Columns after renaming:", df.columns.tolist())
        if 'ds' not in df.columns:
            return {
                "success": False,
                "error_code": "MissingDSColumn",
                "error_message": "Input data must contain a 'ds' column (datetime).",
                "columns": df.columns.tolist()
            }
        df['ds'] = pd.to_datetime(df['ds'])
        if 'y' in df.columns:
            df['y'] = df['y']
        for reg in trained_regressors:
            if reg not in df.columns:
                df[reg] = np.nan
        # Ensure freq is in the form '1H', '1D', etc. for pd.Timedelta
        if freq.isalpha():
            freq_timedelta = "1" + freq
        else:
            freq_timedelta = freq
        if start_datetime is None:
            today_2pm = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
            start_dt = today_2pm + timedelta(hours=36)
        else:
            start_dt = pd.to_datetime(start_datetime)
        if periods is None:
            periods = forecast_days * (24 if freq == 'H' else 1)
        last_ds = df['ds'].max()
        print("last_ds:", last_ds)
        print("start_dt:", start_dt)
        # Prophet's make_future_dataframe starts from the last date in the training data used to fit the model
        model_last_ds = model.history['ds'].max()
        print("model_last_ds:", model_last_ds)
        if hasattr(model_last_ds, 'tzinfo') and model_last_ds.tzinfo is not None:
            model_last_ds = model_last_ds.tz_convert(None)
        if hasattr(start_dt, 'tzinfo') and start_dt.tzinfo is not None:
            start_dt = start_dt.replace(tzinfo=None)
        if start_dt > model_last_ds:
            n_gap = int((start_dt - model_last_ds) / pd.Timedelta(freq_timedelta))
            n_total = n_gap + periods
            future = model.make_future_dataframe(periods=n_total, freq=freq)
            print("future shape before filtering:", future.shape)
            print("future['ds'] min/max:", future['ds'].min(), future['ds'].max())
            future = future[future['ds'] >= start_dt].reset_index(drop=True)
            print("future shape after filtering:", future.shape)
            future = future.iloc[:periods]
        else:
            future = model.make_future_dataframe(periods=periods, freq=freq)
            print("future shape before filtering:", future.shape)
            print("future['ds'] min/max:", future['ds'].min(), future['ds'].max())
            future = future[future['ds'] >= start_dt].reset_index(drop=True)
            print("future shape after filtering:", future.shape)
            future = future.iloc[:periods]
        for reg in trained_regressors:
            if reg in df.columns:
                future[reg] = df[reg].iloc[-1]
            else:
                future[reg] = np.nan
        forecast = model.predict(future)
        forecast_csv_path = "prophet_api_forecast_results.csv"
        forecast.to_csv(forecast_csv_path, index=False)
        return {
            "success": True,
            "forecast_csv": forecast_csv_path,
            "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        }
    except Exception as e:
        return {
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e)
        }

@app.post("/forecast")
def forecast(
    file: UploadFile = File(None),  # Allow file to be optional
    regressors: str = Form(...),
    periods: int = Form(None),
    freq: str = Form("H"),
    start_datetime: str = Form(None),  # New: optional start datetime
    forecast_days: int = Form(6),      # New: number of days to forecast
):
    try:
        if file is not None and hasattr(file, 'file') and file.file is not None:
            file_bytes = file.file.read()
            if file_bytes:
                csv_io = io.BytesIO(file_bytes)
                csv_io.seek(0)
                csv_path = csv_io
            else:
                csv_path = CSV_FILE_PATH
        else:
            csv_path = CSV_FILE_PATH
        result = forecast_with_saved_model_core(
            csv_path=csv_path,
            regressors=regressors,
            periods=periods,
            freq=freq,
            start_datetime=start_datetime,
            forecast_days=forecast_days
        )
        if result["success"]:
            return JSONResponse(result)
        else:
            return JSONResponse(result, status_code=400)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse({
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e),
            "traceback": tb
        }, status_code=500)
# To run: uvicorn prophet_api_forecast:app --reload
