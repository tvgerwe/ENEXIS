import requests

def run_prophet_api(
    csv_path,
    train_start,
    train_end,
    test_start,
    test_end,
    regressors,
    api_url="http://localhost:8000/run-prophet"
):
    """
    Calls the Prophet FastAPI endpoint with the given parameters and CSV file.
    Returns the JSON response from the API.
    """
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
        return response.json()
    except Exception as e:
        return {"success": False, "error_code": type(e).__name__, "error_message": str(e)}

if __name__ == "__main__":
    # Example usage
    csv_path = "src/data/warp-csv-dataset.csv"  # Update path as needed
    train_start = "2025-01-01"
    train_end = "2025-03-14"
    test_start = "2025-03-15"
    test_end = "2025-04-14"
    regressors = "month,shortwave_radiation,apparent_temperature,temperature_2m,direct_normal_irradiance,diffuse_radiation,yearday_sin,Flow_BE,hour_sin,is_non_working_day,is_weekend,is_holiday,weekday_cos,wind_speed_10m,hour_cos,weekday_sin,cloud_cover,Flow_GB,Nuclear_Vol,yearday_cos,Flow_NO,Load"

    result = run_prophet_api(
        csv_path=csv_path,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        regressors=regressors
    )
    print(result)
