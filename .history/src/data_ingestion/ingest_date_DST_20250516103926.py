#!/usr/bin/env python3

import pandas as pd
import numpy as np
import holidays
import sqlite3
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
import pytz

# Zet logging aan
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingest_date')

# Vast pad naar originele .db in repo
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "WARP.db"
# Vervang dit door onderstaande regel als je terug wil naar src/data/
# DB_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "WARP.db"

TABLE_NAME = 'dim_datetime'

def get_connection(db_path):
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database bestaat niet: {db_path}")
    return sqlite3.connect(db_path)

def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None

def get_max_date(conn, table_name):
    if not table_exists(conn, table_name):
        return None
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(datetime) FROM {table_name}")
    max_date = cur.fetchone()[0]
    return pd.to_datetime(max_date) if max_date else None

def create_datetime_rows(start_date, end_date):
    logger.info(f"Creating datetime rows from {start_date} to {end_date}")

    start_date = pd.Timestamp(start_date).tz_localize('UTC') if start_date.tz is None else start_date.tz_convert('UTC')
    end_date = pd.Timestamp(end_date).tz_localize('UTC') if end_date.tz is None else end_date.tz_convert('UTC')

    date_range = pd.date_range(start_date, end_date, freq='h', inclusive='left')
    if len(date_range) == 0:
        logger.info("No new dates to add")
        return None

    df = pd.DataFrame({"datetime": date_range})
    local_tz = pytz.timezone("Europe/Amsterdam")  # ✅ ADD THIS LINE

# Convert UTC datetime to local time
df["datetime_local"] = df["datetime"].dt.tz_convert(local_tz)  # ✅ ADD THIS LINE
df["is_DST"] = df["datetime_local"].apply(lambda x: bool(x.dst()))  # ✅ ADD THIS LINE
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["date"] = df["datetime"].dt.date

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["yearday_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["yearday_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    years = set(df["datetime"].dt.year)
    nl_holidays = holidays.country_holidays("NL", years=list(years))
    df["is_holiday"] = df["date"].isin(nl_holidays).astype(bool)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(bool)
    df["is_non_working_day"] = (df["is_weekend"] | df["is_holiday"]).astype(bool)

    logger.info(f"Created {len(df)} rows")
    return df

def main():
    try:
        logger.info(f"🔍 Using database at: {DB_PATH}")
        conn = get_connection(DB_PATH)

        current_date = pd.Timestamp.now(tz='UTC').floor('h')
        forecast_end = current_date + pd.Timedelta(days=7)

        max_date = get_max_date(conn, TABLE_NAME)

        if max_date is None:
            start_date = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
            logger.info(f"Creating new table from {start_date} to {forecast_end}")
            df = create_datetime_rows(start_date, forecast_end)

            if df is not None:
                logger.info(f"Creating new {TABLE_NAME} table with {len(df)} rows")
                df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_datetime ON {TABLE_NAME}(datetime)")
                conn.commit()
        else:
            if max_date.tz is None:
                max_date = max_date.tz_localize('UTC')

            if max_date < forecast_end:
                new_start = max_date + pd.Timedelta(hours=1)
                logger.info(f"Adding data from {new_start} to {forecast_end}")
                df = create_datetime_rows(new_start, forecast_end)

                if df is not None and not df.empty:
                    logger.info(f"Adding {len(df)} new rows to existing {TABLE_NAME} table")
                    df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
            else:
                logger.info(f"Table {TABLE_NAME} already up to date until {max_date}")

        conn.close()
        logger.info("✅ Datetime dimension update completed")

    except Exception as e:
        logger.error(f"❌ Error updating datetime dimension: {e}", exc_info=True)

if __name__ == "__main__":
    main()