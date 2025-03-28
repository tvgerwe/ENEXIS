from src.data.fetch_weather import fetch_weather_data
from src.features.time_features import add_time_features
import pandas as pd

def main():
    print("🔁 Running ENEXIS pipeline...")

    # Load raw weather data
    df_weather = fetch_weather_data()

    # Add time-based features
    df_features = add_time_features(df_weather)

    # Output preview
    print(df_features.head())

    # Save processed output
    df_features.to_csv("data/processed/weather_with_features.csv", index=False)
    print("✅ Pipeline finished. Output saved to data/processed/")

if __name__ == "__main__":
    main()
