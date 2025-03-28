import pandas as pd
from datetime import datetime

def fetch_weather_data():
    # Simuleer weather data
    df = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", "2022-01-07", freq="H"),
        "temperature": [10 + i % 5 for i in range(24 * 7)],  # dummy data
        "radiation": [200 + i % 50 for i in range(24 * 7)],
    })

    print("🌤️  Weather data fetched (simulated).")
    return df
