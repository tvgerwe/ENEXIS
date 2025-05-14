#!/usr/bin/env python3
# creates 'raw_ned_obs_2025' table, with 1,2,17,20 types

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingest_ned')

# === PAD-INSTELLINGEN ===

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_DB_PATH = PROJECT_ROOT / "src" / "data" / "WARP.db"
LOGS_DB_PATH = PROJECT_ROOT / "src" / "data" / "logs.db"
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.json"

# === CONFIG ===

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config bestand niet gevonden: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# === CONSTANTS ===

TIMESTAMP_COLUMN = 'validfrom' # To be changed to 'validto'  to have right timestamp ?

# === HELPERS ===

def get_connection(db_path):
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database bestaat niet: {db_path}")
    return sqlite3.connect(db_path)

def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None

def ensure_tables_exist(conn_log):
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
    if not table_exists(conn, 'raw_ned_obs_2'):
        return None
    cur = conn.cursor()
    cur.execute(f"SELECT MAX({TIMESTAMP_COLUMN}) FROM raw_ned_obs_2")
    return cur.fetchone()[0]

def create_table_from_df(conn, df, table_name):
    cols = df.columns.tolist()
    col_defs = ', '.join([f'"{col}" TEXT' for col in cols])
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")
    conn.commit()

def remove_duplicates(conn, table_name):
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
            resp = requests.get(endpoint, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            last_url = data.get("hydra:view", {}).get("hydra:last", None)
            members = data.get('hydra:member', [])
            all_recs.extend(members)
            
            if not last_url:
                logger.info(f"Fetched {len(members)} records in single response for type {gen_type}")
            else:
                last_page = int(last_url.split('page=')[-1])
                logger.info(f"Found {last_page} pages for type {gen_type}")
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
                wait_time = 2 ** attempt
                logger.warning(f"Request failed: {e}. Retrying in {wait_time} sec...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise

def write_log(conn, start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message):
    conn.execute(
        """
        INSERT INTO NED_ingestion_log
        (start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (start_time, end_time, api_endpoint, rows_fetched, last_timestamp, status, error_message)
    )
    conn.commit()

# === MAIN LOGICA ===

def main():
    logger.info("🚀 Script started")
    start_time = datetime.datetime.now().isoformat()
    status = 'success'
    error_msg = None
    rows_fetched = 0
    last_timestamp = None
    conn_data = None
    conn_log = None
    endpoint = None

    try:
        logger.info(f"📦 MAIN_DB_PATH: {MAIN_DB_PATH}")
        logger.info(f"📦 LOGS_DB_PATH: {LOGS_DB_PATH}")

        endpoint = config['api']['ned']['endpoint']
        api_key = config['api']['ned']['api_key']
        headers = {'X-AUTH-TOKEN': api_key, 'Content-Type': 'application/json'}
        default_start = config['api']['ned'].get('default_start', config['api']['open_meteo']['default_start'])

        conn_data = get_connection(MAIN_DB_PATH)
        conn_log = get_connection(LOGS_DB_PATH)
        ensure_tables_exist(conn_log)

        #prev_ts = get_last_timestamp(conn_data)
        #start_date = pd.to_datetime(prev_ts).date() if prev_ts else default_start
        start_date = pd.to_datetime("2025-01-01").date()
        end_date = datetime.date.today().isoformat()
        logger.info(f"Startdatum: {start_date} → Einddatum: {end_date}")

        ned_types = config['api']['ned'].get('types')
        
        all_records = []
        for gen_type in ned_types:
            try:
                logger.info(f"gen_type: {gen_type}")
                recs = fetch_records(endpoint, headers, start_date, end_date, gen_type)
                all_records.extend(recs)
            except Exception as e:
                logger.error(f"Error fetching type {gen_type}: {e}")

        if all_records:
            df = pd.DataFrame(all_records)
            df.columns = [c.lower() for c in df.columns]
            if not table_exists(conn_data, 'raw_ned_obs_2'):
                create_table_from_df(conn_data, df, 'raw_ned_obs_2')
                logger.info("Created raw_ned_obs_2 table")

            logger.info(f"Writing {len(df)} records to DB")
            df.to_sql('raw_ned_obs_2', conn_data, if_exists='append', index=False)

            logger.info("Removing duplicates...")
            remove_duplicates(conn_data, 'raw_ned_obs_2')

            rows_fetched = len(df)
            last_timestamp = df[TIMESTAMP_COLUMN].max()
        else:
            last_timestamp = prev_ts or default_start
            logger.info(f"No new records. Last timestamp: {last_timestamp}")

    except Exception as e:
        status = 'failed'
        error_msg = str(e)
        logger.error(f"❌ Process failed: {error_msg}", exc_info=True)

    finally:
        end_time = datetime.datetime.now().isoformat()
        try:
            if conn_log and endpoint:
                write_log(conn_log, start_time, end_time, endpoint, rows_fetched, last_timestamp, status, error_msg)
                logger.info("Log weggeschreven")
        except Exception as log_error:
            logger.error(f"Log wegschrijven mislukt: {log_error}")
        finally:
            if conn_data:
                conn_data.close()
            if conn_log:
                conn_log.close()

if __name__ == '__main__':
    main()