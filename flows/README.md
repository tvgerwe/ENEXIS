# ⚡ Full Pipeline Flow - Electricity Price Forecasting

[![Prefect Flow](https://img.shields.io/badge/flow-prefect-blue)](https://docs.prefect.io/)

This flow automates the full data pipeline for the electricity price forecasting project.

It performs the following steps:

1. **Fetch raw data**  
   - Load ENTSO-E electricity data.
   - Load historical weather data.
   - Generate cyclical time features (sin/cos of hour, day, etc).

2. **Merge data sources**  
   - Combine all sources based on the `datetime` column.

3. **Clean the dataset**  
   - Remove duplicate rows.
   - Fix missing values.
   - Standardize timestamps (UTC).

4. **Validate data**  
   - Check if all required columns are present and correct.

5. **Feature engineering**  
   - Add lag features, rolling averages, calendar features, and scaling.

6. **Split dataset**  
   - Create training and testing datasets (80/20 time split).

---

## 📂 Where does each step happen?

| Step | Location |
|:---|:---|
| Data collection | `src/data_ingestion/` |
| Data merging | `src/data_ingestion/merge_sources.py` |
| Cleaning | `src/data_processing/cleaning.py` |
| Validation | `src/data_processing/validation.py` |
| Feature engineering | `src/data_processing/feature_eng.py` |
| Splitting | `src/data_processing/split.py` |

---

## 🚀 How to run

To run the full pipeline locally:

```bash
python flows/full_pipeline_flow.py
