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

# YOUR ORIGINAL DESIRED COLUMN ORDER (the one that worked!)
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
        # === Load actuals - BUT ONLY THE COLUMNS WE NEED ===
        logger.info("📥 Loading actuals with selected columns only...")
        
        # First check which of our desired columns actually exist in the actuals table
        all_columns_query = f"PRAGMA table_info({ACTUALS_TABLE})"
        available_columns = pd.read_sql_query(all_columns_query, conn)['name'].tolist()
        
        # Filter desired_order to only include columns that exist
        existing_desired_cols = [col for col in desired_order if col in available_columns]
        logger.info(f"📋 Requested columns found: {len(existing_desired_cols)}/{len(desired_order)}")
        logger.info(f"📋 Using columns: {existing_desired_cols}")
        
        # Missing columns
        missing_cols = [col for col in desired_order if col not in available_columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns: {missing_cols}")
        
        # Build the SELECT query with only the columns we want
        columns_str = ", ".join(existing_desired_cols)
        actuals_query = f"SELECT {columns_str} FROM {ACTUALS_TABLE}"
        
        df_actuals = pd.read_sql_query(actuals_query, conn)
        df_actuals["target_datetime"] = pd.to_datetime(df_actuals["target_datetime"], utc=True)
        
        # Filter by date range
        df_actuals = df_actuals[
            (df_actuals["target_datetime"] >= train_start) &
            (df_actuals["target_datetime"] <= train_end)
        ]

        logger.info(f"✅ Actuals loaded: {df_actuals.shape[0]} rows with {df_actuals.shape[1]} selected columns")

        # === Check if we need forecast data ===
        logger.info("🔍 Checking for forecast data...")
        
        # Check if predictions table exists and has data for this run_date
        try:
            pred_count_query = f"""
            SELECT COUNT(*) as count 
            FROM {PREDICTIONS_TABLE} 
            WHERE run_date = '{run_date}'
            AND target_datetime >= '{forecast_start}'
            AND target_datetime <= '{forecast_end}'
            """
            pred_count = pd.read_sql_query(pred_count_query, conn)['count'].iloc[0]
            logger.info(f"📊 Forecast rows available: {pred_count}")
            
            if pred_count > 0:
                logger.info("📊 Found forecast data, but using actuals-only approach for simplicity")
                # You can uncomment the forecast logic below if you want to include forecasts
                
        except Exception as e:
            logger.info(f"📊 No forecast table or data available: {e}")
        
        # === Use actuals only (cleaner approach) ===
        df_combined = df_actuals.copy()
        
        # Sort by datetime and remove duplicates
        df_combined = df_combined.sort_values("target_datetime").drop_duplicates("target_datetime")
        
        # Ensure column order matches desired_order (for columns that exist)
        final_column_order = [col for col in desired_order if col in df_combined.columns]
        df_combined = df_combined[final_column_order]

        logger.info(f"📦 Final table: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns")
        logger.info(f"🧾 Final columns: {df_combined.columns.tolist()}")
        
        # Data quality check
        if 'Price' in df_combined.columns:
            nan_count = df_combined['Price'].isna().sum()
            logger.info(f"💰 Price NaN count: {nan_count}/{len(df_combined)} ({100*nan_count/len(df_combined):.1f}%)")
        
        # Check for any columns with high NaN rates
        high_nan_cols = []
        for col in df_combined.columns:
            if col != 'target_datetime':
                nan_pct = 100 * df_combined[col].isna().sum() / len(df_combined)
                if nan_pct > 20:  # More than 20% NaN
                    high_nan_cols.append(f"{col}: {nan_pct:.1f}%")
        
        if high_nan_cols:
            logger.warning(f"⚠️ Columns with >20% NaN: {high_nan_cols}")
        else:
            logger.info("✅ All columns have good data quality (<20% NaN)")

        # Save to database
        df_combined.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        logger.info(f"✅ Saved as {OUTPUT_TABLE} in {DB_PATH.name}")
        
        return df_combined

    except Exception as e:
        logger.error(f"❌ Error during build: {e}", exc_info=True)
        return None
    finally:
        conn.close()
        logger.info("🔒 Connection closed")


# Optional: Function to add more specific columns if needed
def get_essential_features():
    """
    Returns the essential features for SARIMAX modeling
    """
    return [
        'Price',           # Target variable
        'target_datetime', # Time index
        'Load',           # Main exogenous variable
        'temperature_2m',  # Weather
        'Flow_NO',        # Cross-border flows
        'Flow_GB',        # Cross-border flows
        'hour_cos',       # Time features
        'hour_sin',       # Time features
        'weekday_cos',    # Time features
        'weekday_sin',    # Time features
        'month',          # Seasonal
        'is_weekend',     # Binary features
        'is_holiday'      # Binary features
    ]


def build_minimal_training_set(train_start, train_end, run_date=None):
    """
    Build training set with only essential features for SARIMAX
    """
    essential_cols = get_essential_features()
    
    # Temporarily override desired_order
    global desired_order
    original_order = desired_order.copy()
    desired_order = essential_cols
    
    try:
        result = build_training_set(train_start, train_end, run_date or train_end)
        return result
    finally:
        # Restore original order
        desired_order = original_order


if __name__ == "__main__":
    # Test with your parameters
    result = build_training_set(
        train_start="2025-01-01 00:00:00",
        train_end="2025-03-14 23:00:00",
        run_date="2025-03-15 12:00:00"
    )
    
    if result is not None:
        print(f"\n✅ SUCCESS! Shape: {result.shape}")
        print(f"📋 Columns: {result.columns.tolist()}")
        if 'Price' in result.columns:
            print(f"💰 Price range: {result['Price'].min():.4f} to {result['Price'].max():.4f}")
    else:
        print("❌ FAILED to build training set")