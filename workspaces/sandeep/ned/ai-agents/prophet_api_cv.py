# prophet_api_cv.py
# SUMMARY: FastAPI microservice and core logic for Prophet cross-validation and performance metrics. Provides a core function for running Prophet's cross-validation and metrics, handling file uploads, column normalization, and robust error/debug logging. Used by the unified API and orchestration layer for modular time series validation workflows.

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import io

app = FastAPI()

def cross_validate_model_core(
    csv_path,
    regressors,
    initial,
    period,
    horizon
):
    try:
        import os
        print('Attempting to read CSV from:', csv_path)
        print('File exists?', os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        print("Loaded DataFrame shape:", df.shape)
        print("Loaded DataFrame head:\n", df.head())
        # Always rename 'datetime' to 'ds' if present
        if 'ds' not in df.columns and 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'ds'})
        # Always rename 'Price' to 'y' if present
        if 'y' not in df.columns and 'Price' in df.columns:
            df = df.rename(columns={'Price': 'y'})
        print("Columns after renaming:", df.columns.tolist())
        if 'ds' not in df.columns or 'y' not in df.columns:
            return {
                "success": False,
                "error_code": "MissingDSorYColumn",
                "error_message": "Input data must contain 'ds' (datetime) and 'y' (target) columns.",
                "columns": df.columns.tolist()
            }
        df['ds'] = pd.to_datetime(df['ds'])
        # Remove timezone if present
        if hasattr(df['ds'].dt, 'tz') and df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)
        df['y'] = df['y']
        print('DataFrame shape before fit:', df.shape)
        print('DataFrame columns before fit:', df.columns.tolist())
        print('First few rows before fit:\n', df.head())
        print('Any NaNs in ds?', df['ds'].isna().any())
        print('Any NaNs in y?', df['y'].isna().any())
        regressors_list = [col.strip() for col in regressors.split(',') if col.strip() in df.columns]
        model = Prophet()
        for reg in regressors_list:
            model.add_regressor(reg)
        model.fit(df)
        # Ensure initial, period, and horizon are in the form '1D', '1H', etc.
        def ensure_timedelta_str(val):
            val = str(val)
            if val.isalpha():
                return '1' + val
            return val
        initial = ensure_timedelta_str(initial)
        period = ensure_timedelta_str(period)
        horizon = ensure_timedelta_str(horizon)
        df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        cv_csv_path = "prophet_api_cv_results.csv"
        df_p.to_csv(cv_csv_path, index=False)
        return {
            "success": True,
            "cv_metrics_csv": cv_csv_path
        }
    except Exception as e:
        return {
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e)
        }

@app.post("/cross-validate")
def cross_validate_prophet(
    file: UploadFile = File(...),
    regressors: str = Form(...),
    initial: str = Form(...),
    period: str = Form(...),
    horizon: str = Form(...),
):
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
        # Always rename 'datetime' to 'ds' if present
        if 'ds' not in df.columns and 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'ds'})
        # Always rename 'Price' to 'y' if present
        if 'y' not in df.columns and 'Price' in df.columns:
            df = df.rename(columns={'Price': 'y'})
        print("Columns after renaming:", df.columns.tolist())
        if 'ds' not in df.columns or 'y' not in df.columns:
            return JSONResponse({
                "success": False,
                "error_code": "MissingDSorYColumn",
                "error_message": "Input data must contain 'ds' (datetime) and 'y' (target) columns.",
                "columns": df.columns.tolist()
            }, status_code=400)
        df['ds'] = pd.to_datetime(df['ds'])
        # Remove timezone if present
        if hasattr(df['ds'].dt, 'tz') and df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)
        df['y'] = df['y']
        print('DataFrame shape before fit:', df.shape)
        print('DataFrame columns before fit:', df.columns.tolist())
        print('First few rows before fit:\n', df.head())
        print('Any NaNs in ds?', df['ds'].isna().any())
        print('Any NaNs in y?', df['y'].isna().any())
        regressors_list = [col.strip() for col in regressors.split(',') if col.strip() in df.columns]
        model = Prophet()
        for reg in regressors_list:
            model.add_regressor(reg)
        model.fit(df)
        # Ensure initial, period, and horizon are in the form '1D', '1H', etc.
        def ensure_timedelta_str(val):
            val = str(val)
            if val.isalpha():
                return '1' + val
            return val
        initial = ensure_timedelta_str(initial)
        period = ensure_timedelta_str(period)
        horizon = ensure_timedelta_str(horizon)
        df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        cv_csv_path = "prophet_api_cv_results.csv"
        df_p.to_csv(cv_csv_path, index=False)
        return JSONResponse({
            "success": True,
            "cv_metrics_csv": cv_csv_path
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e)
        }, status_code=500)
# To run: uvicorn prophet_api_cv:app --reload
