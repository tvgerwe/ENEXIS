#!/usr/bin/env python3

import os
import sys
import json
import requests
import pandas as pd
import sqlite3
import datetime
import logging
import time
from pathlib import Path

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MAIN_DB_PATH, LOGS_DB_PATH, config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingest_ned')

# --- Constants ---
TIMESTAMP_COLUMN = 'validfrom'

# --- Helpers ---
def get_connection(db_path):
    """Return a sqlite3 connection."""
    return sqlite3.connect(db_path)

def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None

def ensure_tables_exist(conn_data, conn_log):
    """Create the ingestion log table if it doesn't exist."""
    if not table_exists(conn_log, 'NED_ingestion_log'):
        conn_log.execute("""
            CREATE TABLE NED_ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                end_time TEXT,
                api_endpoint TEXT,
                rows_fetched INTEGER,
                last_timestamp TEXT,
                status TEXT,
                error_message TEXT
            )
        """)
        conn_log.commit()

def get_last_timestamp(conn):
    """Return the maximum value of TIMESTAMP_COLUMN from raw_ned_obs or None if table doesn't exist."""
    if not table_exists(conn, 'raw_ned_obs'):
        return None
    cur = conn.cursor()
    cur.execute(f"SELECT MAX({TIMESTAMP_COLUMN}) FROM raw_ned_obs")
    return cur.fetchone()[0]

def create_table_from_df(conn, df, table_name):
    """Create a new table with columns based on DataFrame."""
    cols = df.columns.tolist()
    col_defs = ', '.join([f'"{col}" TEXT' for col in cols])
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")
    conn.commit()

def remove_duplicates(conn, table_name):
    """Remove exact duplicates from the table, keep the first row per unique set."""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    cols_quoted = ', '.join([f'"{col}"' for col in cols])
    delete_sql = f"""
        DELETE FROM {table_name}
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM {table_name} GROUP BY {cols_quoted}
        )
    """
    conn.execute(delete_sql)
    conn.commit()

def fetch_records(endpoint, headers, start_date, end_date, gen_type, max_retries=3):
    """Paginate through the NED API and collect all records for one gen_type."""
    all_recs = []
    params = {
        'point': '0',
        'type': str(gen_type),
        'granularity': '5',
        'granularitytimezone': '1',
        'classification': '2',
        'activity': '1',
        'validfrom[after]': start_date,
        'validfrom[strictly_before]': end_date,
        'page': 1
    }
    
    logger.info(f"Fetching records for type {gen_type} from {start_date} to {end_date}")
    
    for attempt in range(max_retries):
        try:
            # First request to determine pagination
            resp = requests.get(endpoint, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            last_url = data.get("hydra:view", {}).get("hydra:last", None)
            
            if not last_url:
                # Single page response
                members = data.get('hydra:member', [])
                all_recs.extend(members)
                logger.info(f"Fetched {len(members)} records in single response for type {gen_type}")
            else:    
                # Multi-page response
                last_page = int(last_url.split('page=')[-1])
                logger.info(f"Found {last_page} pages for type {gen_type}")
                
                # Get first page data
                members = data.get('hydra:member', [])
                all_recs.extend(members)
                
                # Process remaining pages
                for page in range(2, last_page + 1):
                    params['page'] = page
                    resp = requests.get(endpoint, params=params, headers=headers, timeout=30)
                    resp.raise_for_status()
                    members = resp.json().get('hydra:member', [])
                    all_recs.extend(members)
                    logger.info(f"Fetched page {page}/{last_page} for type {gen_type}: {len(members)} records")
            
            return all_recs
        
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
    
    return all_recs

def write_log(conn, start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message):
    """Write an entry to the ingestion log."""
    conn.execute(
        """
        INSERT INTO NED_ingestion_log
        (start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message)
    )
    conn.commit()

def main():
    # Initialize variables
    start_time = datetime.datetime.now().isoformat()
    status = 'success'
    error_msg = None
    rows_fetched = 0
    last_timestamp = None
    conn_data = None
    conn_log = None
    endpoint = None

    try:
        # Log the database paths
        logger.info(f"Using main database at: {MAIN_DB_PATH}")
        logger.info(f"Using logs database at: {LOGS_DB_PATH}")
            
        # Get API settings
        endpoint = config['api']['ned']['endpoint']
        api_key = config['api']['ned']['api_key']
        headers = {'X-AUTH-TOKEN': api_key, 'Content-Type': 'application/json'}
        
        # Default start date from config
        default_start = config['api']['ned'].get('default_start', config['api']['open_meteo']['default_start'])
        
        # Connect to DBs
        conn_data = get_connection(MAIN_DB_PATH)
        conn_log = get_connection(LOGS_DB_PATH)
        ensure_tables_exist(conn_data, conn_log)

        # Determine date range
        prev_ts = get_last_timestamp(conn_data)
        
        if prev_ts:
            start_date = pd.to_datetime(prev_ts).date()
            logger.info(f"Continuing from last timestamp: {start_date}")
        else:
            start_date = default_start
            logger.info(f"Starting from default date: {start_date}")
        
        end_date = datetime.date.today().isoformat()

        # Define NED types to fetch
        ned_types = config['api']['ned'].get('types', [2])  # Default to type 2 if not specified

        # Fetch and store
        all_records = []
        for gen_type in ned_types:
            try:
                recs = fetch_records(endpoint, headers, start_date, end_date, gen_type)
                all_records.extend(recs)
                logger.info(f"Successfully fetched {len(recs)} records for type {gen_type}")
            except Exception as e:
                logger.error(f"Error fetching type {gen_type}: {e}")
                # Continue with other types instead of failing completely

        if all_records:
            df = pd.DataFrame(all_records)
            df.columns = [c.lower() for c in df.columns]
            if not table_exists(conn_data, 'raw_ned_obs'):
                create_table_from_df(conn_data, df, 'raw_ned_obs')
                logger.info("Created new raw_ned_obs table")
            
            logger.info(f"Writing {len(df)} records to database")
            df.to_sql('raw_ned_obs', conn_data, if_exists='append', index=False)
            
            logger.info("Removing duplicates")
            remove_duplicates(conn_data, 'raw_ned_obs')
            
            rows_fetched = len(df)
            last_timestamp = df[TIMESTAMP_COLUMN].max()
            logger.info(f"Successfully processed {rows_fetched} records. Latest timestamp: {last_timestamp}")
        else:
            last_timestamp = prev_ts or default_start
            logger.info(f"No new records fetched. Last timestamp remains: {last_timestamp}")

    except Exception as e:
        status = 'failed'
        error_msg = str(e)
        logger.error(f"Process failed: {error_msg}", exc_info=True)
    finally:
        end_time = datetime.datetime.now().isoformat()
        
        try:
            if conn_log and endpoint:
                write_log(conn_log, start_time, end_time, endpoint, rows_fetched, last_timestamp, status, error_msg)
                logger.info("Wrote to ingestion log")
        except Exception as log_error:
            logger.error(f"Failed to write to log: {log_error}")
            
        try:
            if conn_data:
                conn_data.close()
            if conn_log:
                conn_log.close()
            logger.info("Closed database connections")
        except Exception as close_error:
            logger.error(f"Error closing connections: {close_error}")

if __name__ == '__main__':
    main()
