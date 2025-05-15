# env - enexis-may-03-env-run

import os
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


def prepare_data(df):
    print("inside prepare_data")
    df = df[df['Price'] > 0].copy()
    df = df.rename(columns={'datetime': 'ds', 'Price': 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    print("Finished prepare_data")
    return df


def build_and_fit_model(df, changepoint_prior_scale=0.001, seasonality_mode='multiplicative', seasonality_prior_scale=10.0):
    print("inside build_and_fit_model")

    # Identify regressors (optional for now, could be added with add_regressor)
    regressors = [col for col in df.columns if col not in ['y', 'ds']]

    # Time series split (latest fold only)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(df):
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)

    print(f"Train Date Range: {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"Test Date Range:  {test_df['ds'].min()} to {test_df['ds'].max()}")

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode
    )

    # Optional: add regressors if using external variables
    # for reg in regressors:
    #     model.add_regressor(reg)

    model.fit(train_df)
    print("Finished build_and_fit_model")
    return model


def run_cross_validation(model):
    print("Running cross-validation...")
    df_cv = cross_validation(
        model,
        initial='90 days', # 365 days
        period='300 days', # 90 days
        horizon='7 days', # 180 days
        parallel="processes"  # use "threads" if "processes" gives issues
    )
    print("Cross-validation complete.")
    return df_cv


def evaluate(df_cv):
    print("Evaluating metrics...")
    df_p = performance_metrics(df_cv)
    print(df_p[['horizon', 'mae', 'rmse', 'mape', 'smape']].head())
    return df_p


def plot_forecast(model, forecast):
    model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.tight_layout()
    plt.show()

    model.plot_components(forecast)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    db_path = '/Users/sgawde/work/eaisi-code/main-branch-11-may/ENEXIS/src/data/WARP.db'
    conn = sqlite3.connect(db_path)
    df_pd_orig = pd.read_sql_query("SELECT * FROM master_warp ORDER BY datetime DESC", conn)
    conn.close()

    df_pd_orig['datetime'] = pd.to_datetime(df_pd_orig['datetime'])
    df = df_pd_orig.sort_values(by='datetime')

    # Prepare data
    df_prepared = prepare_data(df)

    # Fit model
    start = time.time()
    model = build_and_fit_model(df_prepared)
    print("Training complete in", round(time.time() - start, 2), "seconds.")

    # Cross-validation
    df_cv = run_cross_validation(model)

    # Metrics
    df_metrics = evaluate(df_cv)

    # Forecast
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)

    # Plot
    plot_forecast(model, forecast)

    print("Model run complete")
