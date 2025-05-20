import requests
import os
import joblib
import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_PATH = "/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/workspaces/sandeep/ned/downloads/prophet_ai-agent-model.pkl"
REGRESSORS_PATH = "/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/workspaces/sandeep/ned/downloads/prophet_ai-agent-model-regressors.txt"


@app.post("/forecast")
def forecast(
    file: UploadFile = File(...),
    regressors: str = Form(...),
    periods: int = Form(...),
    freq: str = Form("D"),
):
    try:
        if not os.path.exists(MODEL_PATH):
            return JSONResponse({
                "success": False,
                "error_code": "ModelNotFound",
                "error_message": f"Trained model not found at {MODEL_PATH}"
            }, status_code=500)
        model = joblib.load(MODEL_PATH)
        # Load the regressor list used during training
        if os.path.exists(REGRESSORS_PATH):
            with open(REGRESSORS_PATH, "r") as f:
                trained_regressors = [line.strip() for line in f if line.strip()]
        else:
            trained_regressors = [col.strip() for col in regressors.split(",") if col.strip()]
        df = pd.read_csv(io.BytesIO(file.file.read()))
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y']
        # Ensure all regressors used in training are present in the DataFrame
        for reg in trained_regressors:
            if reg not in df.columns:
                df[reg] = np.nan
        # Reorder columns: ds, y, then regressors in the same order as training
        ordered_cols = ['ds', 'y'] + trained_regressors
        df = df.reindex(columns=ordered_cols)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        for reg in trained_regressors:
            last_val = df[reg].iloc[-1] if not df[reg].isnull().all() else 0
            future[reg] = list(df[reg]) + [last_val]*periods
        # Reorder future columns as well
        future = future.reindex(columns=['ds'] + trained_regressors)
        forecast = model.predict(future)
        forecast_csv_path = "prophet_api_forecast.csv"
        forecast.to_csv(forecast_csv_path, index=False)
        return JSONResponse({
            "success": True,
            "forecast_csv": forecast_csv_path
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse({
            "success": False,
            "error_code": type(e).__name__,
            "error_message": str(e),
            "traceback": tb
        }, status_code=500)

def run_prophet_forecast_api(
    csv_path,
    regressors,
    periods,
    freq="D",
    api_url="http://localhost:8001/forecast"
):
    """
    Calls the Prophet Forecast FastAPI endpoint with the given parameters and CSV file.
    Returns the JSON response from the API.
    """
    with open(csv_path, "rb") as f:
        files = {"file": ("data.csv", f, "text/csv")}
        data = {
            "regressors": regressors,
            "periods": periods,
            "freq": freq
        }
        response = requests.post(api_url, files=files, data=data)
    try:
        return response.json()
    except Exception as e:
        return {"success": False, "error_code": type(e).__name__, "error_message": str(e)}

if __name__ == "__main__":
    # Example usage
    csv_path = "src/data/warp-csv-dataset.csv"  # Update path as needed
    regressors = "month,shortwave_radiation,apparent_temperature,temperature_2m,direct_normal_irradiance,diffuse_radiation,yearday_sin,Flow_BE,hour_sin,is_non_working_day,is_weekend,is_holiday,weekday_cos,wind_speed_10m,hour_cos,weekday_sin,cloud_cover,Flow_GB,Nuclear_Vol,yearday_cos,Flow_NO,Load"
    periods = 30
    result = run_prophet_forecast_api(
        csv_path=csv_path,
        regressors=regressors,
        periods=periods
    )
    print(result)
