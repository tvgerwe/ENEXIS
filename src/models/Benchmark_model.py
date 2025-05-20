# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 🛠️ Voorkom OverflowError bij te veel punten
mpl.rcParams['agg.path.chunksize'] = 10000

# 📂 Bestanden
forecast_path = "/Users/redouan/Downloads/Price_Preds_Processed_20250407.csv"
actuals_path = "/Users/redouan/Downloads/GUI_ENERGY_PRICES_202501010000-202601010000.csv"

# 🧠 Load & transform forecast data
forecast_df = pd.read_csv(forecast_path)
origin = pd.Timestamp("1992-02-18")
forecast_df["forecast_date"] = origin + pd.to_timedelta(forecast_df["x"], unit="m")
forecast_df["forecast_date_only"] = forecast_df["forecast_date"].dt.normalize()

# 🧠 Load & transform actuals
actuals_df = pd.read_csv(actuals_path, encoding='utf-8-sig')
actuals_df["date_anchor"] = actuals_df["MTU (UTC)"].astype(str).str.split(" - ").str[1]
actuals_df["actual_date"] = pd.to_datetime(actuals_df["date_anchor"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
actuals_df["price_kwh"] = actuals_df["Day-ahead Price (EUR/MWh)"] / 1000
actuals_df["actual_date_only"] = actuals_df["actual_date"].dt.normalize()

# 🔗 Merge
merged_df = pd.merge(
    forecast_df,
    actuals_df,
    left_on="forecast_date_only",
    right_on="actual_date_only",
    how="inner"
)

# 🧽 Schoonmaken
merged_clean = merged_df.dropna(subset=["price_kwh", "y"])

if merged_clean.empty:
    print("❌ Geen geldige data om te vergelijken.")
    exit(1)

# 📊 Evaluatie
y_true = merged_clean["price_kwh"]
y_pred = merged_clean["y"]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan))) * 100
mase_scale = np.mean(np.abs(y_true.diff().dropna()))
mase = mae / mase_scale if mase_scale != 0 else np.nan

print(f"✅ RMSE: {rmse}")
print(f"✅ MAE: {mae}")
print(f"✅ MAPE (%): {mape}")
print(f"✅ MASE: {mase}")

# 📈 Visualisatie
plt.figure(figsize=(14, 6))
plt.plot(merged_clean["forecast_date_only"], y_true, label="Werkelijke prijs (kWh)", linewidth=1.5)
plt.plot(merged_clean["forecast_date_only"], y_pred, label="Voorspelde prijs (kWh)", linestyle="--", linewidth=1.5)
plt.title("⚡ Elektriciteitsprijs: Voorspelling vs Realiteit")
plt.xlabel("Datum")
plt.ylabel("Prijs (EUR/kWh)")
plt.legend()
plt.grid(True)

# 📌 Metrics tonen als tekst op de figuur
metrics_text = (
    f"RMSE: {rmse:.5f}\n"
    f"MAE: {mae:.5f}\n"
    f"MAPE: {mape:.2f}%\n"
    f"MASE: {mase:.2f}"
)

plt.gcf().text(0.75, 0.7, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()