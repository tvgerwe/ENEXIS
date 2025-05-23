import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time

API_KEY = '82aa28d4-59f3-4e3a-b144-6659aa9415b5'
domain = '10YNL----------L'  # Nederland
base_url = "https://web-api.tp.entsoe.eu/api"

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 3)  # voorbeeld: pas aan voor jouw periode

all_forecasts = []

for delta in range((end_date - start_date).days):
    target_day = start_date + timedelta(days=delta)
    period_start = target_day.strftime('%Y%m%d%H%M')
    period_end = (target_day + timedelta(days=1)).strftime('%Y%m%d%H%M')
    for days_ahead in range(1, 8):  # 1 t/m 7 dagen vooruit
        publication_date = target_day - timedelta(days=days_ahead)
        pub_start = publication_date.strftime('%Y%m%d%H%M')
        pub_end = (publication_date + timedelta(days=1)).strftime('%Y%m%d%H%M')
        params = {
            'securityToken': API_KEY,
            'documentType': 'A65',  # Load forecast
            'processType': 'A01',   # Day-ahead
            'outBiddingZone_Domain': domain,
            'periodStart': period_start,
            'periodEnd': period_end,
            'publication_MarketDocument.period.timeInterval.start': pub_start,
            'publication_MarketDocument.period.timeInterval.end': pub_end
        }
        print(f"Ophalen forecast voor {target_day.date()} gepubliceerd op {publication_date.date()} ({days_ahead} dagen vooruit)")
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Geen data voor {target_day.date()} ({days_ahead} dagen vooruit)")
            time.sleep(1)
            continue
        try:
            root = ET.fromstring(response.content)
            # Zoek alle tijdsblokken
            for timeseries in root.findall('.//{*}TimeSeries'):
                for pt in timeseries.findall('.//{*}Point'):
                    position = int(pt.find('{*}position').text)
                    quantity = float(pt.find('{*}quantity').text)
                    # Bepaal tijdstip
                    dt = target_day + timedelta(hours=position-1)
                    all_forecasts.append({
                        'target_datetime': dt,
                        'forecast_MW': quantity,
                        'forecast_date': publication_date.date(),
                        'days_ahead': days_ahead
                    })
        except Exception as e:
            print(f"Fout bij parsen: {e}")
        time.sleep(1)  # rate limiting

df = pd.DataFrame(all_forecasts)
print(df.head())
df.to_csv('historical_weekahead_forecasts.csv', index=False)
print("Klaar! Data opgeslagen in 'historical_weekahead_forecasts.csv'")