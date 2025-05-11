#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import holidays
from datetime import date, datetime, timedelta
import sqlite3
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('create_dim_datetime')

# Constants
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / 'config' / 'config.json'
TABLE_NAME = 'dim_datetime'

def load_config():
    """Load JSON config from file."""
    import json
    if not CONFIG_PATH.exists():
        alt_paths = [
            ROOT_DIR / 'src' / 'config' / 'config.json',
            Path('/Users/redouan/ENEXIS/config/config.json'),
            Path('/Users/redouan/ENEXIS/src/config/config.json'),
            Path(os.getcwd()) / 'config' / 'config.json',
            Path(os.getcwd()) / 'src' / 'config' / 'config.json'
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                return json.load(open(alt_path, 'r'))
        
        # Default configuration if no config file found
        return {
            "database": {
                "main_db_path": "src/data/WARP.db"
            }
        }
    
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def get_connection(db_path):
    """Return a sqlite3 connection, create folder if needed."""
    folder = os.path.dirname(db_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    return sqlite3.connect(db_path)

def table_exists(conn, table_name):
    """Check if a table exists in the database."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None

def get_max_date(conn, table_name):
    """Get the maximum date in the datetime table."""
    if not table_exists(conn, table_name):
        return None
    
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(datetime) FROM {table_name}")
    max_date = cur.fetchone()[0]
    
    if max_date:
        return pd.to_datetime(max_date)
    return None

def create_datetime_rows(start_date, end_date):
    """
    Create datetime dimension rows for the specified date range
    
    Parameters:
    start_date (datetime): Start date for new rows
    end_date (datetime): End date for new rows
    
    Returns:
    pd.DataFrame: DataFrame with datetime features
    """
    logger.info(f"Creating datetime rows from {start_date} to {end_date}")
    
    # Make sure both dates are UTC for consistent handling
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_convert('UTC')
    else:
        start_date = pd.Timestamp(start_date).tz_localize('UTC')
        
    if hasattr(end_date, 'tz') and end_date.tz is not None:
        end_date = end_date.tz_convert('UTC')
    else:
        end_date = pd.Timestamp(end_date).tz_localize('UTC')
    
    # Generate hourly timestamps
    date_range = pd.date_range(start_date, end_date, freq='h', inclusive='left')
    if len(date_range) == 0:
        logger.info("No new dates to add")
        return None
    
    time_df = pd.DataFrame({"datetime": date_range})
    
    # Basic time components needed for SARIMAX
    time_df["hour"] = time_df["datetime"].dt.hour
    time_df["day_of_week"] = time_df["datetime"].dt.dayofweek
    time_df["month"] = time_df["datetime"].dt.month
    time_df["day_of_year"] = time_df["datetime"].dt.dayofyear
    time_df["date"] = time_df["datetime"].dt.date
    
    # Cyclical encoding - essential for time series models
    time_df["hour_sin"] = np.sin(2 * np.pi * time_df["hour"] / 24)
    time_df["hour_cos"] = np.cos(2 * np.pi * time_df["hour"] / 24)
    time_df["weekday_sin"] = np.sin(2 * np.pi * time_df["day_of_week"] / 7)
    time_df["weekday_cos"] = np.cos(2 * np.pi * time_df["day_of_week"] / 7)
    time_df["yearday_sin"] = np.sin(2 * np.pi * time_df["day_of_year"] / 365.25)
    time_df["yearday_cos"] = np.cos(2 * np.pi * time_df["day_of_year"] / 365.25)
    
    # Calendar features highly relevant for energy prices
    # Determine Dutch holidays
    years_set = set(time_df["datetime"].dt.year)
    nl_holidays = holidays.country_holidays("NL", years=list(years_set))
    time_df["is_holiday"] = time_df["date"].isin(nl_holidays).astype('bool')
    time_df["is_weekend"] = time_df["day_of_week"].isin([5, 6]).astype('bool')
    time_df["is_non_working_day"] = (time_df["is_weekend"] | time_df["is_holiday"]).astype('bool')
    
    logger.info(f"Created {len(time_df)} rows")
    return time_df

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Get database path
        db_path = Path(config['database']['main_db_path'])
        if not db_path.is_absolute():
            db_path = ROOT_DIR / db_path
        
        # Debug path issues
        data_dir = db_path.parent
        logger.info(f"Using database at: {db_path}")
        logger.info(f"Data directory: {data_dir}, exists: {data_dir.exists()}")
        
        # Make sure directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Connect to database
        conn = get_connection(str(db_path))
        
        # Determine date range for updates
        # Use UTC timezone for all datetime operations to be consistent
        current_date = pd.Timestamp.now(tz='UTC').floor('H')
        forecast_end = current_date + pd.Timedelta(days=7)  # Current time + 7 days
        
        # Check if table exists and get max date
        max_date = get_max_date(conn, TABLE_NAME)
        
        if max_date is None:
            # Table doesn't exist or is empty - create from 2025
            start_date = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
            logger.info(f"Creating new table from {start_date} to {forecast_end}")
            time_df = create_datetime_rows(start_date, forecast_end)
            
            if time_df is not None:
                logger.info(f"Creating new {TABLE_NAME} table with {len(time_df)} rows")
                time_df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
                
                # Create index for faster lookups
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_datetime ON {TABLE_NAME}(datetime)")
                conn.commit()
        else:
            # Table exists - only add new rows if needed
            # Ensure max_date has timezone info
            if max_date.tz is None:
                max_date = max_date.tz_localize('UTC')
                
            if max_date < forecast_end:
                # Need to add more rows
                new_start = max_date + pd.Timedelta(hours=1)  # Start from next hour
                logger.info(f"Adding data from {new_start} to {forecast_end}")
                time_df = create_datetime_rows(new_start, forecast_end)
                
                if time_df is not None and not time_df.empty:
                    logger.info(f"Adding {len(time_df)} new rows to existing {TABLE_NAME} table")
                    time_df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
            else:
                logger.info(f"Table {TABLE_NAME} already up to date until {max_date}")
        
        # Close connection
        conn.close()
        logger.info("Datetime dimension update completed")
        
    except Exception as e:
        logger.error(f"Error updating datetime dimension: {e}", exc_info=True)

if __name__ == "__main__":
    main()