import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - build_training_set - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_training_set")

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "WARP.db"
OUTPUT_TABLE = "training_set"
ACTUALS_TABLE = "master_warp"
PREDICTIONS_TABLE = "master_predictions"
HORIZON = 168
FORECAST_LAG_HOURS = 36 
FORECAST_HORIZON_HOURS = 144

# Gewenste kolomvolgorde (Price als target vooraan)
desired_order = [
    'Price', 'target_datetime', 'Load', 'shortwave_radiation', 'temperature_2m',
    'direct_normal_irradiance', 'diffuse_radiation', 'Flow_NO', 'yearday_cos',
    'Flow_GB', 'month', 'is_dst', 'yearday_sin', 'is_non_working_day',
    'hour_cos', 'is_weekend', 'cloud_cover', 'weekday_sin', 'hour_sin',
    'weekday_cos'
]

def build_training_set(train_start, train_end, run_date):
    train_start = pd.Timestamp(train_start, tz="UTC")
    train_end = pd.Timestamp(train_end, tz="UTC")
    run_date = pd.Timestamp(run_date, tz="UTC")
    forecast_start = run_date
    forecast_end = forecast_start + pd.Timedelta(hours=HORIZON - 1)

    logger.info("🚀 Start build van trainingset")
    logger.info(f"🧠 Actuals van {train_start} t/m {train_end}")
    logger.info(f"📅 Forecast van run_date {run_date}, target range: {forecast_start} → {forecast_end}")

    conn = sqlite3.connect(DB_PATH)

    try:
        # === Load actuals
        df_actuals = pd.read_sql_query(f"SELECT * FROM {ACTUALS_TABLE}", conn)
        df_actuals["target_datetime"] = pd.to_datetime(df_actuals["target_datetime"], utc=True)
        df_actuals = df_actuals[
            (df_actuals["target_datetime"] >= train_start) &
            (df_actuals["target_datetime"] <= train_end)
        ]

        # Kolommen die weg mogen
        columns_to_exclude = ['wind_direction_10m', 'direct_radiation', 'Price_actual', 'datetime', 'date']
        keep_columns = [col for col in df_actuals.columns if col not in columns_to_exclude]
        df_actuals = df_actuals[keep_columns]
        logger.info(f"✅ Actuals geladen: {df_actuals.shape[0]} rijen")

        # === Load forecast features
        df_preds = pd.read_sql_query(f"SELECT * FROM {PREDICTIONS_TABLE}", conn)
        df_preds["target_datetime"] = pd.to_datetime(df_preds["target_datetime"], utc=True)
        df_preds["run_date"] = pd.to_datetime(df_preds["run_date"], utc=True)

        df_preds = df_preds[
            (df_preds["run_date"] == run_date) &
            (df_preds["target_datetime"] >= forecast_start) &
            (df_preds["target_datetime"] <= forecast_end)
        ]

        if df_preds.columns.duplicated().any():
            dupes = df_preds.columns[df_preds.columns.duplicated()].tolist()
            logger.warning(f"⚠️ Dubbele kolomnamen in df_preds: {dupes}")
            df_preds = df_preds.loc[:, ~df_preds.columns.duplicated()]

        # === Get actual prices for forecast period from master_warp
        df_forecast_prices = pd.read_sql_query(f"""
            SELECT target_datetime, Price 
            FROM {ACTUALS_TABLE} 
            WHERE target_datetime >= '{forecast_start}' 
            AND target_datetime <= '{forecast_end}'
        """, conn)
        
        if not df_forecast_prices.empty:
            df_forecast_prices["target_datetime"] = pd.to_datetime(df_forecast_prices["target_datetime"], utc=True)
            
            # Merge actual prices into forecast features
            df_preds = df_preds.merge(df_forecast_prices, on="target_datetime", how="left")
            logger.info(f"✅ Added actual prices to {len(df_forecast_prices)} forecast rows")
        else:
            # Add Price column as NaN if no actuals found
            df_preds['Price'] = pd.NA
            logger.warning(f"⚠️ No actual prices found for forecast period")

        # Ensure both dataframes have the same columns
        all_columns = set(df_actuals.columns) | set(df_preds.columns)
        
        # Add missing columns to both dataframes
        for col in all_columns:
            if col not in df_actuals.columns:
                df_actuals[col] = pd.NA
            if col not in df_preds.columns:
                df_preds[col] = pd.NA

        # Reorder columns to match
        common_columns = sorted(all_columns)
        df_actuals = df_actuals[common_columns]
        df_preds = df_preds[common_columns]

        # Combine actuals + forecasts
        df_combined = pd.concat([df_actuals, df_preds], ignore_index=True)
        df_combined = df_combined.sort_values("target_datetime").drop_duplicates("target_datetime")

        # Force column order
        available_cols = [col for col in desired_order if col in df_combined.columns]
        remaining_cols = [col for col in df_combined.columns if col not in desired_order]
        df_combined = df_combined[available_cols + remaining_cols]

        logger.info(f"📦 Eindtabel bevat: {df_combined.shape[0]} rijen, {df_combined.shape[1]} kolommen")
        logger.info(f"🧾 Kolommen: {df_combined.columns.tolist()}")
        
        # Show Price distribution
        nan_count = df_combined['Price'].isna().sum()
        logger.info(f"❓ Price NaN count: {nan_count}/{len(df_combined)} ({100*nan_count/len(df_combined):.1f}%)")

        # Save to database AND return DataFrame
        df_combined.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ Opgeslagen als {OUTPUT_TABLE} in {DB_PATH.name}")
        
        return df_combined

    except Exception as e:
        logger.error(f"❌ Fout tijdens build: {e}", exc_info=True)
        return None
    finally:
        conn.close()
        logger.info("🔒 Verbinding gesloten")