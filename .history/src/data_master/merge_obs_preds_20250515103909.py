def combine_observations_and_forecasts(df_forecast, df_weather):
    """Voegt voorspellingen en observaties samen."""
    df_weather_combined = pd.merge(df_forecast, df_weather, on="datetime", how="outer")

    for col in df_weather.columns:
        if col != "datetime":
            col_forecast = f"{col}_x"
            col_obs = f"{col}_y"
            if col_forecast in df_weather_combined and col_obs in df_weather_combined:
                df_weather_combined[col] = df_weather_combined[col_obs].combine_first(
                    df_weather_combined[col_forecast]
                )
    return df_weather_combined