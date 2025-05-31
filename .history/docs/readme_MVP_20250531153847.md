# ⚡ WARP - Electricity Price Forecasting MVP

## 📟 Project Overview
"Week-Ahead Reduced Pricing for Electricity Costs" (WARP) is a minimal, functional system that provides **hourly electricity price forecasts** for the Netherlands with a **7-day horizon**. Its accuracy is assessed by the average RMSE of all 144 predicted hourly prices, in each iteration (prediction). We compare the model performance with a benchmaker model. 

---

## 📦 Final Deliverables

- A working forecasting pipeline and model
- Transparent documentation on performance, implications and future recommendations (this report)

---

## 🔄 1. Data Pipeline & Automation

A modular ETL system for:

- Collecting external data from:
  - [ENTSO-E](https://www.entsoe.eu/) (market prices)
  - [Open-Meteo](https://open-meteo.com/) (weather parameters)
  - [The Oxygent](https://energie.theoxygent.nl/) (electricity price prediction benchmark)
- Cleaning, validating, and harmonizing time series inputs

---

## 🤖 2. Core Forecasting Engine

A forecasting model (e.g. **SARIMA** / **RandomForest** / **XGBoost** / **Prophet**) that:

- Predicts hourly electricity prices **1 to 7 days ahead**
- Is optimized for **lowest RMSE across 6 days prediction periods**
- Trains on:
  - Historical prices and cross-border (electricity) flow (ENTSO-E)
  - Weather data (temperature, wind, solar radiation, cloud cover)
- Compares performance using **RMSE**, against baseline & benchmark models

---

## 🧪 3. Codebase and Documentation

Delivered in a version-controlled GitHub repository:

- `README.md` – Setup instructions, usage guide, input/output explanation
- `requirements.txt` – Python dependencies
- `.env` – API tokens or config (excluded from version control)
- Scripts:
  - `training_test.py` – Combined historical and predictive data, in (+1 day) rolling forecast approach
  - `<<Model Name>> evaluation.ipynb` – Notebook per forecasting model, including 30 runs per feature selection setting
- Modular file structure: separation between **data**, **model**, **evaluation**, and **visualization** python and notebook files



