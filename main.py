from src.data.fetch_weather import fetch_weather_data
from src.data.fetch_entsoe import fetch_entsoe_data
from src.features.time_features import add_time_features
from src.models.baseline import train_naive_model

def main():
    print("🔁 Running ENEXIS pipeline...")

    df_weather = fetch_weather_data()
    df_weather_features = add_time_features(df_weather)

    df_energy = fetch_entsoe_data()

    df_energy_features = add_time_features(df_energy)  # Je kunt ook hier tijdfeatures toepassen

    _, metrics = train_naive_model(df_energy_features)

    print(f"✅ Pipeline finished. MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}")

if __name__ == "__main__":
    main()
