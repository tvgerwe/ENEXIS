# %% [markdown]
# # Time Series Forecasting: Naive Model with Plotly Visualization (ENTSOE Dataset)
#
# This notebook uses your ENTSOE energy prices dataset as in the SARIMA model.
# We load and preprocess the data from the provided file path, split it into training and testing sets,
# build a naive forecast (using the last observation from the training set), and visualize the results with Plotly.
#
# We use your existing data loading code for the ENTSOE dataset.
# We will adjust the file path or file name when the data pipe line is set.

# %%
import sys

import os
import pandas as pd
import numpy as np

# Define file path components (adjust file name as needed)
DATA_FOLDER = "/Users/redouan/Downloads/ENTSOE"
FILE_NAME = "GUI_ENERGY_PRICES_202212312300-202312312300.csv"
FILE_PATH = os.path.join(DATA_FOLDER, FILE_NAME)

# Check if file exists
if not os.path.exists(FILE_PATH):
    print(f"❌ Error: File not found at {FILE_PATH}")
    sys.exit(1)

print(f"✅ File found: {FILE_PATH}")

# Load the raw dataset and print its shape and columns
df_raw = pd.read_csv(FILE_PATH)
print(f"✅ Raw data shape: {df_raw.shape}")
print("✅ Raw data columns:", df_raw.columns.tolist())
print("✅ Raw data preview:")
print(df_raw.head())

# Extract the start time from the "MTU (CET/CEST)" column.
# The column is in the format: "dd/mm/yyyy HH:MM:SS - dd/mm/yyyy HH:MM:SS"
df_raw["Timestamp"] = df_raw["MTU (CET/CEST)"].str.split(" - ").str[0]

# Parse the "Timestamp" column using the explicit format
df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
print("\n✅ Timestamps after parsing:")
print(df_raw["Timestamp"].head())

# Drop any rows with invalid timestamps (NaT)
df_raw = df_raw.dropna(subset=["Timestamp"])
print(f"\n✅ Data shape after dropping invalid timestamps: {df_raw.shape}")

# Set the "Timestamp" column as the index and sort the DataFrame
df_raw.set_index("Timestamp", inplace=True)
df_raw.sort_index(inplace=True)
print("\n✅ Index range after sorting:")
print(f"   Start: {df_raw.index.min()}")
print(f"   End:   {df_raw.index.max()}")

# Drop duplicate timestamps (keep the first occurrence)
df_raw = df_raw[~df_raw.index.duplicated(keep="first")]
print("\n✅ Data shape after dropping duplicate timestamps:", df_raw.shape)

# Rename the energy price column:
# Use the "Day-ahead (EUR/MWh)" column as the target variable.
if "Day-ahead (EUR/MWh)" in df_raw.columns:
    df_raw.rename(columns={"Day-ahead (EUR/MWh)": "Energy Price"}, inplace=True)
else:
    print("⚠️ Warning: 'Day-ahead (EUR/MWh)' column not found.")

# Keep only the "Energy Price" column
if "Energy Price" in df_raw.columns:
    df_processed = df_raw[["Energy Price"]]
else:
    print("❌ ERROR: 'Energy Price' column is missing. Check column names.")
    sys.exit(1)

# Ensure the index has a continuous hourly frequency (filling missing timestamps via forward fill)
df_processed = df_processed.asfreq("H").ffill()
print("\n✅ Final preprocessed data shape:", df_processed.shape)
print("✅ Final index range:")
print(f"   Start: {df_processed.index.min()}")
print(f"   End:   {df_processed.index.max()}")

# Assign the processed DataFrame to 'df' for use in subsequent steps
df = df_processed

# %% [markdown]
#
# We define the training and testing periods as per your original code:
# - Training period: 2023-09-01 to 2023-10-21 (≈3 weeks)
# - Test period: 2023-10-22 to 2023-10-28 (≈1 week)
#
# We then check for data leakage to ensure the training data ends before the test data begins.

# %%
# Define training and testing date ranges (adjust as needed)
train_start = "2023-09-01"
train_end = "2023-10-21"   # 3 weeks of training data
test_start  = "2023-10-22"
test_end    = "2023-10-28"   # 1 week of test/forecasting data

# Filter the DataFrame to select the training and test periods
train = df.loc[train_start:train_end]
test = df.loc[test_start:test_end]

# Check for data leakage: ensure that the training set ends before the test set begins
if train.index.max() < test.index.min():
    print("✅ No data leakage: Training data ends before Test data begins.")
else:
    raise ValueError("❌ Data leakage detected: Training data overlaps with Test data!")

# Print summary of the train-test split
print("\n✅ Train Data Summary:")
print(f"   Start: {train.index.min()}, End: {train.index.max()}, Total: {len(train)} records")
print("✅ Test Data Summary:")
print(f"   Start: {test.index.min()}, End: {test.index.max()}, Total: {len(test)} records")

# %% [markdown]
# ## Naive Forecast Model
#
# The naive forecasting model uses the last observed value from the training set as the forecast for every timestamp in the test set.
# This simple baseline is useful when comparing against more advanced models like SARIMA.

# %%
def naive_forecast(train_df, test_df):
    """
    Generate a naive forecast using the last observation in the training set.
    
    Parameters:
        train_df (DataFrame): Training data containing a 'Energy Price' column.
        test_df (DataFrame): Test data for forecasting.
    
    Returns:
        np.array: Array with the forecasted values for the test set.
    """
    last_value = train_df["Energy Price"].iloc[-1]
    forecast_values = np.repeat(last_value, len(test_df))
    return forecast_values

# Generate the forecast using the naive model
forecast = naive_forecast(train, test)

# %% [markdown]
# ## Forecast Evaluation & Visualization with Plotly
#
# We evaluate the forecast using Mean Absolute Error (MAE) and visualize the training data, test data, and naive forecast.
# The Plotly figure is set to use the "plotly_white" template for a white background.

# %%
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from math import sqrt
import numpy as np

# Calculate the MAE
mae = mean_absolute_error(test["Energy Price"], forecast)
print("Mean Absolute Error (MAE):", mae)

# Calculate additional metrics

# RMSE: Root Mean Squared Error
rmse = sqrt(np.mean((test["Energy Price"] - forecast)**2))

# MAPE: Mean Absolute Percentage Error
# We protect against division by zero with np.where.
mape = np.mean(np.abs((test["Energy Price"] - forecast) / np.where(test["Energy Price"] != 0, test["Energy Price"], 1))) * 100

# MASE: Mean Absolute Scaled Error
# Scale is based on the average absolute one-step change in the training set.
scale = np.mean(np.abs(train["Energy Price"].diff().dropna()))
mase = mae / scale if scale != 0 else np.nan

# AIC: Akaike Information Criterion approximation.
# Note: AIC is traditionally derived from likelihood-based models, so this is a rough estimate.
n = len(test)
rss = np.sum((test["Energy Price"] - forecast)**2)
aic = n * np.log(rss / n) + 2  # using k = 1 for the constant forecast

print("RMSE:", rmse)
print("MAPE:", mape)
print("MASE:", mase)
print("AIC:", aic)

# Create a Plotly figure
fig = go.Figure()

# Add training data trace
fig.add_trace(go.Scatter(
    x=train.index, y=train["Energy Price"],
    mode="lines", name="Training Data"
))

# Add test data trace
fig.add_trace(go.Scatter(
    x=test.index, y=test["Energy Price"],
    mode="lines", name="Test Data"
))

# Add naive forecast trace
fig.add_trace(go.Scatter(
    x=test.index, y=forecast,
    mode="lines", name="Naive Forecast", line=dict(dash="dash")
))

# Update layout with a white background
fig.update_layout(
    title="Naive Forecast vs Actual Energy Prices",
    xaxis_title="Timestamp",
    yaxis_title="Energy Price (EUR/MWh)",
    template="plotly_white"
)

# Add an annotation box with the metrics details
metrics_text = (
    f"MAE: {mae:.2f}<br>"
    f"RMSE: {rmse:.2f}<br>"
    f"MAPE: {mape:.2f}%<br>"
    f"MASE: {mase:.2f}<br>"
    f"AIC: {aic:.2f}"
)

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.05, y=0.95,
    text=metrics_text,
    showarrow=False,
    bgcolor="white",
    bordercolor="black"
)

fig.show()

# %% [markdown]
# ## Conclusion
#
# The naive forecast leverages the last observed energy price in the training period to predict all future values.
# While overly simplistic, it is a crucial baseline for assessing the performance of more complex models (e.g., SARIMA).
#
# **Data Note:**
# This notebook uses your ENTSOE energy prices CSV file. If you switch to a different dataset, please adjust
# the file path, data-loading, and preprocessing steps as necessary.
# %%
