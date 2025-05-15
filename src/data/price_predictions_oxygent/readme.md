# 📈 Forecast Dataset – WARP Project

This dataset contains electricity price forecasts used in the **WARP** project (Week-Ahead Reduced Pricing for Electricity Costs), a data science initiative by the EAISI Cohort (Nov 2024) and ENEXIS.

## 🔍 Overview

Each row in this dataset represents a set of electricity price predictions fetched from the dynamic pricing API, along with metadata about when and how the data was retrieved and processed.

| Column            | Description |
|-------------------|-------------|
| `x`               | Raw timestamp in minutes since origin (`1992-02-18`), used to calculate forecasted datetime. |
| `y`               | Forecasted price in EUR/kWh (excluding taxes). |
| `timestamp`       | UTC timestamp when the data was fetched from the API. |
| `subarray`        | Position in forecast sequence (0 = published price, >0 = hour-ahead forecast steps). |
| `date_timestamp`  | Date (UTC) extracted from `timestamp`. |
| `hour_timestamp`  | Hour extracted from `timestamp`, rounded down to the full hour. |
| `date_time`       | UTC datetime derived from `x`; represents the hour the price applies to. |
| `Price`           | Final consumer price in EUR/kWh including VAT and energy tax, derived from `y`. |

## 📁 File Origin

- Data fetched via notebooks:
  - `get_prices.ipynb`: API fetching
  - `process_prices.ipynb`: transformation steps
- Source format: CSV with numeric, datetime, and derived metadata fields.

## 🧠 Notes

- The dataset is used to train and evaluate models predicting future electricity prices.
- Subarray >0 entries indicate predictions made for hours beyond the moment of publishing.
- The price signal (`y`) is converted to a consumer-facing rate (`Price`) using national energy tax policies.


CSV files contains fetched price predictions, obtained up till the date mentioned.
x y and subarray are values directly obtained from the API
the code for that is in workspaces - twan, and is named get_prices_ipynb
other collumns are added, step by step, as coded in proces_prices.ipynb
which is located in the same folder.
there additional collumns are:
timestamp = UTC timestamp of moment of fetch from API
date_timestamp = date deducted from 'timestamp'
hour_timestamp = hour deducted from 'timestamp'. round down to full hour
date_time = UTC datetime conversion from 'x', showing the hour for which price point is predicted (or published in case of subarray == 0)
Price = electricity price including VAT and energy tax. obtained through conversion from 'y' value. 
