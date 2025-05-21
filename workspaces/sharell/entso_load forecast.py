import pandas as pd
from entsoe import EntsoePandasClient
from datetime import timedelta
import time

API_KEY = '82aa28d4-59f3-4e3a-b144-6659aa9415b5'
client = EntsoePandasClient(api_key=API_KEY)
country_code = 'NL'

start_date = pd.Timestamp('2025-01-01', tz='Europe/Amsterdam')
end_date = pd.Timestamp('2025-05-21', tz='Europe/Amsterdam')  # exclusief

all_forecasts = []

current_start = start_date
while current_start < end_date:
    current_end = current_start + timedelta(days=1)
    try:
        print(f"Ophalen forecast van {current_start.date()} tot {current_end.date()}")
        # Haal week-ahead forecast op, dit geeft meestal 7 dagen vooruit
        forecast = client.query_load_forecast(country_code, start=current_start, end=current_end)
        # forecast is al een DataFrame of Series, check datatype
        if isinstance(forecast, pd.Series):
            df = forecast.to_frame(name='forecast_MW')
        else:
            df = forecast.copy()  # als het al DataFrame is
        
        df['forecast_date'] = current_start.date()
        df['target_datetime'] = df.index
        all_forecasts.append(df.reset_index(drop=True))
    except Exception as e:
        print(f"Fout bij {current_start.date()}: {e}")
    time.sleep(1)  # voorkom rate limiting
    current_start = current_end

# Alles samenvoegen
result = pd.concat(all_forecasts, ignore_index=True)
print(result.head(20))

# Opslaan naar CSV
result.to_csv('weekahead_forecast_jan_2024.csv', index=False)
print("Klaar! Week-ahead forecast opgeslagen in 'weekahead_forecast_jan_2024.csv'")
