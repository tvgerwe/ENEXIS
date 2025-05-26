import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = "src/models/model_run_results/forecast_vs_actual.csv"
print("File exists:", os.path.exists(csv_file))

df = pd.read_csv(csv_file)
print(df.head())
print(df.columns)

# If the first column is a date, parse it
if 'date' in df.columns[0].lower():
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

df.plot(x=df.columns[0], y=df.columns[1:], kind='line', marker='o')
plt.title("Line Chart from CSV Data")
plt.xlabel(df.columns[0])
plt.ylabel("Values")
plt.grid(True)
plt.tight_layout()
plt.show()