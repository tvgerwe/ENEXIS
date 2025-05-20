# prophet_api.py
# SUMMARY: FastAPI microservice and core logic for Prophet model build and training. Provides a core function for model training, hyperparameter tuning, and artifact saving, as well as an API endpoint for file upload and model build. Used by the unified API and orchestration layer for modular time series workflows.

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import io
import time
import joblib
from sklearn.model_selection import ParameterGrid

app = FastAPI()

def build_and_save_prophet_model_core(
    csv_path,
    train_start,
    train_end,
    test_start,
    test_end,
    regressors
):
    try:
        df = pd.read_csv(csv_path)
        print("Loaded DataFrame shape:", df.shape)
        print("Loaded DataFrame head:\n", df.head())
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
        df = df.sort_values(by='datetime')
        y = df[['datetime', 'Price']]
        X = df.drop(columns=['Price'])
        X_train = X[(X['datetime'] >= train_start) & (X['datetime'] <= train_end)].copy()
        X_test  = X[(X['datetime'] >= test_start) & (X['datetime'] <= test_end)].copy()
        y_train = y[(y['datetime'] >= train_start) & (y['datetime'] <= train_end)].copy()
        y_test  = y[(y['datetime'] >= test_start) & (y['datetime'] <= test_end)].copy()
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        regressors_list = [col.strip() for col in regressors.split(',') if col.strip() in X_train.columns]
        train_prophet = pd.concat([
            y_train[['datetime', 'Price']].reset_index(drop=True),
            X_train[regressors_list].reset_index(drop=True)
        ], axis=1)
        test_prophet = pd.concat([
            y_test[['datetime', 'Price']].reset_index(drop=True),
            X_test[regressors_list].reset_index(drop=True)
        ], axis=1)
        print("train_prophet shape:", train_prophet.shape)
        print("test_prophet shape:", test_prophet.shape)
        train_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
        test_prophet.rename(columns={'datetime': 'ds', 'Price': 'y'}, inplace=True)
        train_prophet['ds'] = pd.to_datetime(train_prophet['ds']).dt.tz_localize(None)
        test_prophet['ds'] = pd.to_datetime(test_prophet['ds']).dt.tz_localize(None)
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [1.0, 5.0, 10.0],
            'holidays_prior_scale': [1.0, 5.0, 10.0]
        }
        best_rmse = float('inf')
        best_params = None
        best_model = None
        best_forecast = None
        for params in ParameterGrid(param_grid):
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale']
            )
            for reg in regressors_list:
                model.add_regressor(reg)
            model.fit(train_prophet)
            forecast = model.predict(test_prophet)
            merged = test_prophet.set_index('ds')[['y']].join(forecast.set_index('ds')[['yhat']], how='inner').dropna()
            rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model
                best_forecast = forecast
        model_save_path = "prophet-ai-agent-model.pkl"
        joblib.dump(best_model, model_save_path)
        predictions_csv_path = "prophet_api_predictions.csv"
        merged = test_prophet.set_index('ds')[['y']].join(best_forecast.set_index('ds')[['yhat']], how='inner').dropna()
        merged.reset_index().to_csv(predictions_csv_path, index=False)
        metrics_csv_path = "prophet_api_metrics.csv"
        metrics_df = pd.DataFrame([{
            "RMSE": best_rmse,
            "BestParams": best_params
        }])
        metrics_df.to_csv(metrics_csv_path, index=False)
        response = {
            "success": True,
            "RMSE": best_rmse,
            "best_params": best_params,
            "model_path": model_save_path,
            "predictions_csv": predictions_csv_path,
            "metrics_csv": metrics_csv_path
        }
        return response
    except Exception as e:
        return {
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e)
        }

@app.post("/run-prophet")
def run_prophet(
    file: UploadFile = File(...),
    train_start: str = Form(...),
    train_end: str = Form(...),
    test_start: str = Form(...),
    test_end: str = Form(...),
    regressors: str = Form(...),  # comma-separated string
):
    try:
        # Save uploaded file to a temporary path
        temp_file_path = "temp_uploaded_file.csv"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())

        # Call the core function
        response = build_and_save_prophet_model_core(
            csv_path=temp_file_path,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            regressors=regressors
        )

        # Return response
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e)
        }, status_code=500)

# To run: uvicorn prophet_api:app --reload
