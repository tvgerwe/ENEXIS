import streamlit as st
import pandas as pd
import plotly.express as px
import logging
from pathlib import Path
import json

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prophet_rolling_validation')

# === Config Setup ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "sandeep" / "config" / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_RUN_RESULTS_DIR = config['ned']['ned_model_download_dir']
model_file_path = f'{MODEL_RUN_RESULTS_DIR}prophet_model.pkl'
rolling_window_file_path = f'{MODEL_RUN_RESULTS_DIR}rolling_validation_results.csv'
forecast_output_path = f"{MODEL_RUN_RESULTS_DIR}/forecast_vs_actual.csv"

# Load data
df_metrics = pd.read_csv(rolling_window_file_path)
df_forecast = pd.read_csv(forecast_output_path)

st.title("📈 Prophet Forecast Validation Dashboard")

# Show metrics
st.subheader("🧮 Model Performance by Window")
st.dataframe(df_metrics)

# Select window
selected_window = st.selectbox("Select Train Window End Date", df_metrics["window_end"].astype(str).unique())

# Filter actual vs predicted
filtered = df_forecast[df_forecast["train_window_end"].astype(str) == selected_window]

# Plot
fig = px.line(filtered, x='date', y=['actual', 'predicted'], title=f'Forecast vs Actual ({selected_window})')
st.plotly_chart(fig)
