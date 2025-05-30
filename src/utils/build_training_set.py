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
    'Flow_GB', 'month', 'is_dst', 'yearday_sin','wind_speed_10m', 'is_non_working_day',
    'hour_cos', 'is_weekend', 'cloud_cover', 'weekday_sin', 'hour_sin',
    'weekday_cos'
]

def build_training_set(train_start, train_end, run_date, lag_hours=168):
    train_start = pd.Timestamp(train_start, tz="UTC")
    train_end = pd.Timestamp(train_end, tz="UTC")
    run_date = pd.Timestamp(run_date, tz="UTC")
    # Normalize run_date to midnight since that's how it's stored in the DB
    run_date_normalized = run_date.normalize()  # This sets time to 00:00:00
    forecast_start = run_date
    forecast_end = forecast_start + pd.Timedelta(hours=HORIZON)
    
    # Check if we have enough historical data for lagging
    earliest_lag_needed = forecast_end - pd.Timedelta(hours=lag_hours)
    if earliest_lag_needed < train_start:
        logger.warning(f"⚠️ Lagging issue detected!")
        logger.warning(f"   Latest prediction: {forecast_end}")
        logger.warning(f"   Needs lag data from: {earliest_lag_needed} ({lag_hours}h lag)")
        logger.warning(f"   But training starts: {train_start}")
        logger.warning(f"   Consider extending train_start or reducing forecast horizon")
    
    # Don't extend training end - keep it as specified by user
    # We'll handle lagging data needs separately
    original_train_end = train_end
    
    # Check if we have enough historical data for lagging
    earliest_lag_needed = forecast_end - pd.Timedelta(hours=lag_hours)
    if earliest_lag_needed < train_start:
        logger.warning(f"⚠️ Lagging issue detected!")
        logger.warning(f"   Latest prediction: {forecast_end}")
        logger.warning(f"   Needs lag data from: {earliest_lag_needed} ({lag_hours}h lag)")
        logger.warning(f"   But training starts: {train_start}")
        logger.warning(f"   Consider extending train_start or reducing forecast horizon")
    
    # For lagging purposes, we need to load additional historical data
    # but we won't include it in the final training set
    required_data_end = max(original_train_end, earliest_lag_needed)
    if required_data_end > original_train_end:
        logger.info(f"📅 Loading additional historical data until {required_data_end} for lagging support")
        extended_train_end = required_data_end
    else:
        extended_train_end = original_train_end

    logger.info("🚀 Start build van trainingset")
    logger.info(f"🧠 Actuals van {train_start} t/m {original_train_end} (extended to {extended_train_end} for lagging)")
    logger.info(f"📅 Forecast van run_date {run_date}, normalized to {run_date_normalized} for DB lookup, target range: {forecast_start} → {forecast_end}")

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
        
        # Load extended data for lagging purposes
        df_actuals_extended = df_actuals[
            (df_actuals["target_datetime"] >= train_start) &
            (df_actuals["target_datetime"] <= extended_train_end)
        ]
        
        # Keep only the original training period for final output
        df_actuals = df_actuals[
            (df_actuals["target_datetime"] >= train_start) &
            (df_actuals["target_datetime"] <= original_train_end)
        ]

        logger.info(f"✅ Actuals loaded: {df_actuals.shape[0]} rows with {df_actuals.shape[1]} selected columns")

        # === NOW ACTUALLY LOAD AND USE THE FORECAST DATA ===
        logger.info("🔍 Loading forecast/prediction data...")
        df_predictions = None
        
        try:
            # Check if predictions table exists and has data for this run_date
            pred_count_query = f"""
            SELECT COUNT(*) as count 
            FROM {PREDICTIONS_TABLE} 
            WHERE run_date = '{run_date_normalized}'
            AND target_datetime >= '{forecast_start}'
            AND target_datetime <= '{forecast_end}'
            """
            pred_count = pd.read_sql_query(pred_count_query, conn)['count'].iloc[0]
            logger.info(f"📊 Forecast rows available: {pred_count}")
            
            if pred_count > 0:
                # Check which columns exist in predictions table
                pred_columns_query = f"PRAGMA table_info({PREDICTIONS_TABLE})"
                pred_available_columns = pd.read_sql_query(pred_columns_query, conn)['name'].tolist()
                
                # Find common columns between actuals and predictions (excluding run_date which might only be in predictions)
                common_cols = [col for col in existing_desired_cols if col in pred_available_columns]
                logger.info(f"📋 Common columns for predictions: {len(common_cols)} - {common_cols}")
                
                if common_cols:
                    # Load predictions with same column structure as actuals
                    pred_columns_str = ", ".join(common_cols)
                    predictions_query = f"""
                    SELECT {pred_columns_str}
                    FROM {PREDICTIONS_TABLE} 
                    WHERE run_date = '{run_date_normalized}'
                    AND target_datetime >= '{forecast_start}'
                    AND target_datetime <= '{forecast_end}'
                    ORDER BY target_datetime
                    """
                    
                    df_predictions = pd.read_sql_query(predictions_query, conn)
                    df_predictions["target_datetime"] = pd.to_datetime(df_predictions["target_datetime"], utc=True)
                    
                    logger.info(f"✅ Predictions loaded: {df_predictions.shape[0]} rows with {df_predictions.shape[1]} columns")
                    
                    # === HANDLE MISSING COLUMNS WITH 168-HOUR LAG ===
                    missing_cols = [col for col in existing_desired_cols if col not in df_predictions.columns]
                    logger.info(f"🔧 Missing columns in predictions: {missing_cols}")
                    
                    if missing_cols:
                        logger.info(f"📊 Applying {lag_hours}-hour lag for missing columns (excluding target variables)...")
                        
                        # Define columns that should NOT be lagged
                        no_lag_columns = {'Price', 'target_datetime'}  # Price is target, target_datetime is time index
                        
                        # For each missing column, find the values from 168 hours ago
                        for col in missing_cols:
                            if col in no_lag_columns:
                                # Don't lag target variables - leave as NaN
                                df_predictions[col] = None
                                logger.info(f"   🎯 Column '{col}' is target variable - filled with NaN (not lagged)")
                            elif col in df_actuals.columns:
                                logger.info(f"   🕐 Lagging column '{col}' by {lag_hours} hours")
                                
                                # Create lagged values for each prediction timestamp
                                lagged_values = []
                                for pred_time in df_predictions['target_datetime']:
                                    # Find the value lag_hours earlier
                                    lag_time = pred_time - pd.Timedelta(hours=lag_hours)
                                    
                                    # Look for this timestamp in extended actuals (for lagging)
                                    matching_actual = df_actuals_extended[df_actuals_extended['target_datetime'] == lag_time]
                                    
                                    if not matching_actual.empty:
                                        lagged_values.append(matching_actual[col].iloc[0])
                                    else:
                                        # If no exact match, find the closest earlier timestamp
                                        earlier_actuals = df_actuals_extended[df_actuals_extended['target_datetime'] <= lag_time]
                                        if not earlier_actuals.empty:
                                            closest_actual = earlier_actuals.iloc[-1]  # Most recent before lag_time
                                            lagged_values.append(closest_actual[col])
                                            logger.debug(f"     📅 {pred_time}: used {closest_actual['target_datetime']} instead of {lag_time}")
                                        else:
                                            lagged_values.append(None)
                                            logger.warning(f"     ❌ {pred_time}: no lagged data available for {lag_time}")
                                
                                # Add the lagged column to predictions
                                df_predictions[col] = lagged_values
                                
                                non_null_count = sum(1 for v in lagged_values if v is not None)
                                logger.info(f"   ✅ Added {col}: {non_null_count}/{len(lagged_values)} values found")
                            else:
                                # For columns not in actuals, fill with None
                                df_predictions[col] = None
                                logger.info(f"   🔧 Added missing column '{col}' (filled with NaN)")
                    
                    # Reorder columns to match actuals
                    df_predictions = df_predictions[existing_desired_cols]
                else:
                    logger.warning("⚠️ No common columns found between actuals and predictions tables!")
                    
        except Exception as e:
            logger.warning(f"📊 Could not load predictions: {e}")
        
        # === COMBINE ACTUALS AND PREDICTIONS ===
        if df_predictions is not None and not df_predictions.empty:
            logger.info("🔄 Combining actuals and predictions...")
            
            # === RETRIEVE ACTUAL PRICES FOR FORECAST PERIOD ===
            logger.info("💰 Retrieving actual prices for forecast period...")
            
            try:
                # Get actual prices for the forecast period from master_warp
                forecast_actuals_query = f"""
                SELECT target_datetime, Price 
                FROM {ACTUALS_TABLE}
                WHERE target_datetime >= '{forecast_start}'
                AND target_datetime <= '{forecast_end}'
                ORDER BY target_datetime
                """
                
                df_forecast_actuals = pd.read_sql_query(forecast_actuals_query, conn)
                df_forecast_actuals['target_datetime'] = pd.to_datetime(df_forecast_actuals['target_datetime'], utc=True)
                
                logger.info(f"📊 Found {len(df_forecast_actuals)} actual prices for forecast period")
                
                if not df_forecast_actuals.empty:
                    # Create a mapping of datetime -> actual price
                    price_mapping = dict(zip(df_forecast_actuals['target_datetime'], df_forecast_actuals['Price']))
                    
                    # Fill in actual prices for predictions where available
                    actual_prices_filled = 0
                    for idx, row in df_predictions.iterrows():
                        pred_time = row['target_datetime']
                        if pred_time in price_mapping:
                            df_predictions.loc[idx, 'Price'] = price_mapping[pred_time]
                            actual_prices_filled += 1
                    
                    logger.info(f"✅ Filled {actual_prices_filled}/{len(df_predictions)} prediction prices with actual values")
                    
                    # Show price coverage
                    non_null_prices = df_predictions['Price'].notna().sum()
                    logger.info(f"💰 Price coverage: {non_null_prices}/{len(df_predictions)} ({100*non_null_prices/len(df_predictions):.1f}%)")
                else:
                    logger.warning("⚠️ No actual prices found for forecast period - prices will remain NaN")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve forecast period prices: {e}")
            
            df_combined = pd.concat([df_actuals, df_predictions], ignore_index=True)
            logger.info(f"✅ Combined dataset: {df_combined.shape[0]} rows ({df_actuals.shape[0]} actuals + {df_predictions.shape[0]} predictions)")
        else:
            logger.info("📊 No predictions to combine, using actuals only")
            df_combined = df_actuals.copy()
        
        # Sort by datetime and handle overlaps properly
        df_combined = df_combined.sort_values("target_datetime")
        
        # Check for overlaps between actuals and predictions
        if df_predictions is not None and not df_predictions.empty:
            overlap_start = max(df_actuals['target_datetime'].min(), df_predictions['target_datetime'].min())
            overlap_end = min(df_actuals['target_datetime'].max(), df_predictions['target_datetime'].max())
            
            if overlap_start <= overlap_end:
                logger.info(f"⚠️ Overlap detected between actuals and predictions: {overlap_start} → {overlap_end}")
                logger.info("   Keeping actuals for overlapping periods, predictions for non-overlapping periods")
                
                # For overlapping timestamps, keep actuals; for non-overlapping, keep predictions
                df_combined = df_combined.drop_duplicates("target_datetime", keep='first')
            else:
                logger.info("✅ No overlap between actuals and predictions")
        else:
            # Remove duplicates within actuals only
            df_combined = df_combined.drop_duplicates("target_datetime", keep='first')
        
        # Ensure column order matches desired_order (for columns that exist)
        final_column_order = [col for col in desired_order if col in df_combined.columns]
        df_combined = df_combined[final_column_order]

        logger.info(f"📦 Final combined table: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns")
        logger.info(f"🧾 Final columns: {df_combined.columns.tolist()}")
        
        # Show date range
        if not df_combined.empty:
            min_date = df_combined['target_datetime'].min()
            max_date = df_combined['target_datetime'].max()
            logger.info(f"📅 Date range: {min_date} → {max_date}")
        
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
    # Test with your parameters - using a run_date that exists in your DB
    result = build_training_set(
        train_start="2025-01-01 00:00:00",
        train_end="2025-03-14 23:00:00",  # End day before your first prediction run_date
        run_date="2025-03-15 00:00:00"   # This will be normalized to 2025-05-14 00:00:00 for DB lookup
    )
    
    if result is not None:
        print(f"\n✅ SUCCESS! Shape: {result.shape}")
        print(f"📋 Columns: {result.columns.tolist()}")
        if 'Price' in result.columns:
            print(f"💰 Price range: {result['Price'].min():.4f} to {result['Price'].max():.4f}")
        
        # Show breakdown of data
        if 'target_datetime' in result.columns:
            train_end_ts = pd.Timestamp("2025-03-14 23:00:00", tz="UTC")
            actuals_count = (result['target_datetime'] <= train_end_ts).sum()
            predictions_count = (result['target_datetime'] > train_end_ts).sum()
            print(f"📊 Breakdown: {actuals_count} actuals + {predictions_count} predictions")
    else:
        print("❌ FAILED to build training set")