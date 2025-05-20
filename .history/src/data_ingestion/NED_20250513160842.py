#!/usr/bin/env python3

import os
import json
import requests
import pandas as pd
import sqlite3
import datetime

# --- Paths & Constants ---
# Base directory (two levels up from this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Config file path
CONFIG_PATH = os.path.join(ROOT_DIR, 'workspaces', 'sandeep', 'config', 'api-call.json')
# Data database (existing)
WARP_DB_PATH = os.path.join(ROOT_DIR, 'src', 'data', 'WARP.db')
# Logs database (will be created if not exists)
LOGS_DB_PATH = os.path.join(ROOT_DIR, 'src', 'data', 'logs.db')

# Initial date range for first run (fixed)
DEFAULT_START_DATE = '2022-01-01'
DEFAULT_END_DATE = '2025-01-01'
# API types to fetch (adjust as needed)
NED_TYPES = [1,2,17,20] #4 relevant types: 1,2,17,20
# Name of timestamp column (will lowercase all df columns)
TIMESTAMP_COLUMN = 'validfrom'

# --- Helpers ---
def load_config(path):
    """Load JSON config from file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_connection(db_path):
    """Return a sqlite3 connection."""
    return sqlite3.connect(db_path)


def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def ensure_tables_exist(conn_data, conn_log):
    """Ensure required tables exist in both databases."""
    # raw data table will be created dynamically upon first write
    # Ensure ingestion log table in logs.db
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
    """Return the maximum timestamp from raw_ned_df, or None if table absent."""
    if not table_exists(conn, 'raw_ned_df'):
        return None
    cur = conn.cursor()
    cur.execute(f"SELECT MAX({TIMESTAMP_COLUMN}) FROM raw_ned_df")
    return cur.fetchone()[0]


def create_table_from_df(conn, df, table_name):
    """Create a new table with all DataFrame columns as TEXT."""
    cols = df.columns.tolist()
    col_defs = ', '.join([f'"{col}" TEXT' for col in cols])
    sql = f'CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})'
    conn.execute(sql)
    conn.commit()


def remove_duplicates(conn, table_name):
    """Remove exact duplicate rows, keeping first occurrence."""
    cur = conn.cursor()
    # fetch column names
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


def fetch_records(endpoint, headers, start_date, end_date, gen_type):
    """Paginate through the NED API and collect all records."""
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
    # First request to get last page
    resp = requests.get(endpoint, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    last_url = data.get('hydra:view', {}).get('hydra:last')
    if not last_url:
        raise RuntimeError('Could not determine last page from API response')
    last_page = int(last_url.split('page=')[-1])

    # Loop through all pages
    for page in range(1, last_page + 1):
        params['page'] = page
        resp = requests.get(endpoint, params=params, headers=headers)
        resp.raise_for_status()
        members = resp.json().get('hydra:member', [])
        all_recs.extend(members)
    return all_recs


def write_log(conn, start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message):
    """Insert a record into the ingestion log."""
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
    start_time = datetime.datetime.now().isoformat()
    status = 'success'
    error_msg = None
    rows_fetched = 0
    last_timestamp = None

    try:
        # Load config
        config = load_config(CONFIG_PATH)
        endpoint = config['ned']['ned_api_endpoint']
        api_key = config['ned']['demo-ned-api-key']
        headers = {
            'X-AUTH-TOKEN': api_key,
            'Content-Type': 'application/json'
        }

        # Connect to DBs
        conn_data = get_connection(WARP_DB_PATH)
        conn_log = get_connection(LOGS_DB_PATH)
        ensure_tables_exist(conn_data, conn_log)

        # Determine date range
        prev_ts = get_last_timestamp(conn_data)
        if prev_ts:
            start_date = prev_ts
            end_date = datetime.date.today().isoformat()
        else:
            start_date = DEFAULT_START_DATE
            end_date = DEFAULT_END_DATE

        # Fetch data
        all_records = []
        for gen_type in NED_TYPES:
            recs = fetch_records(endpoint, headers, start_date, end_date, gen_type)
            all_records.extend(recs)

        # Insert into raw_ned_df
        if all_records:
            df = pd.DataFrame(all_records)
            # normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Create table if first run
            if not table_exists(conn_data, 'raw_ned_df'):
                create_table_from_df(conn_data, df, 'raw_ned_df')

            df.to_sql('raw_ned_df', conn_data, if_exists='append', index=False)
            remove_duplicates(conn_data, 'raw_ned_df')

            rows_fetched = len(df)
            last_timestamp = df[TIMESTAMP_COLUMN].max()
        else:
            rows_fetched = 0
            last_timestamp = prev_ts or start_date

    except Exception as e:
        status = 'failed'
        error_msg = str(e)
    finally:
        end_time = datetime.datetime.now().isoformat()
        write_log(conn_log, start_time, end_time, endpoint, rows_fetched, last_timestamp, status, error_msg)
        # Close connections
        conn_data.close()
        conn_log.close()


if __name__ == '__main__':
    main()
