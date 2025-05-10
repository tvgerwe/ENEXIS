# 📦 What needs to happen, step-by-step

---

## 1. Data Collection (split into separate scripts)

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Collect ENTSO-E data | `src/data_ingestion/entsoe.py` | `data/raw/entsoe_raw.csv` | Use `entsoe-py` to retrieve load, prices, and cross-border flows, and save them as CSV. |
| Collect Weather data | `src/data_ingestion/weather.py` | `data/raw/weather_raw.csv` | Use Open-Meteo API to retrieve historical temperature and radiation data. |
| Create Sin/Cos time features | `src/data_ingestion/sin_cos.py` | `data/raw/time_features.csv` | Create cyclic time columns such as hour_sin, dayofyear_cos, etc. |

---

## 2. Merge source data

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Merge sources into one dataset | `src/data_ingestion/merge_sources.py` | `data/processed/full_dataset.csv` | Merge ENTSO-E, Weather, and Sin/Cos DataFrames based on the `datetime` column. |

---

## 3. Create Cleaning Code

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Clean the data | `src/data_processing/cleaning.py` | `data/processed/full_dataset_cleaned.csv` | Remove duplicates, convert timezone to UTC, fill missing values (`fillna(method='ffill')`). |

---

## 4. Add Validation

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Validate the data | `src/data_processing/validation.py` | Validation log or error messages | Check for columns, data types, logical value checks (`assert`), and NaNs. |

---

## 5. Expand Feature Engineering

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Apply feature engineering | `src/data_processing/feature_eng.py` | `data/final/final_dataset.csv` | Add lag features, rolling averages, time features, and scaling to the dataset. |

---

## 6. Clean Train/Test Split

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Time-based data split | `src/data_processing/split.py` | `data/final/train.csv`, `data/final/test.csv` | Perform a chronological 80% train and 20% test split without randomization. |

---

## 7. Restructure Models

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Place time series models | `src/models/time_series/` | Model files (.pkl or fitted models) | Train ARIMA or other time series models. |
| Place machine learning models | `src/models/ml_models/` | Model files (.pkl) | Train ML models like Random Forest or Prophet. |

---

## 8. Separate Predictions and Evaluations

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Create evaluation metrics and plots | `src/evaluation/metrics.py` | Plots, evaluation scores | Calculate RMSE/MAE, generate comparison plots of forecast vs actuals. |
| Store results | outputs/ | `outputs/forecast_results.csv` | Save forecasts, errors, and summaries as CSV files. |

---

## 9. Move Reporting and Exploration

| Step | File | Output | Explanation |
|:---|:---|:---|:---|
| Structure exploratory analysis and modeling notebooks | `notebooks/exploratory/`, `notebooks/modeling/` | Jupyter notebooks with EDA and model development | Create separate notebooks for exploratory analysis and model training. |

---

# ✨ Summary:
- Each data source gets its own extraction script.
- Sources are merged first, then processed.
- Each operation (cleaning, validation, feature engineering) has its own clear script.
- Models and outputs are neatly organized.
- This structure supports scaling, maintenance, and clarity.

---
