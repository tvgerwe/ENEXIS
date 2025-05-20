#!/usr/bin/env python3
import os
import json
import time
import datetime
import sqlite3
import pandas as pd
from entsoe import EntsoePandasClient

ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH  = os.path.join(ROOT_DIR, 'workspaces', 'sandeep', 'config', 'api-call.json')
WARP_DB_PATH = os.path.join(ROOT_DIR, 'src', 'data', 'WARP.db')
TABLE_RAW    = 'raw_entsoe_obs'


def load_config(path):
    with open(path) as f:
        return json.load(f)


def get_connection(db_path):
    folder = os.path.dirname(db_path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    return sqlite3.connect(db_path)


def table_exists(conn, name):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def create_table_from_df(conn, df, name):
    cols = df.columns.tolist()
    defs = ','.join(f'"{c}" TEXT' for c in cols)
    conn.execute(f'CREATE TABLE IF NOT EXISTS {name} ({defs})')
    conn.commit()


def remove_duplicates(conn, name):
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info({name})')
    cols = [r[1] for r in cur.fetchall()]
    group = ','.join(f'"{c}"' for c in cols)
    conn.execute(
        f'DELETE FROM {name} WHERE rowid NOT IN ('
        f'SELECT MIN(rowid) FROM {name} GROUP BY {group})'
    )
    conn.commit()


def fetch_with_retries(func, *args, retries=3, delay=5, **kwargs):
    last_exc = None
    for _ in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch after {retries} retries") from last_exc


def ingest_entsoe():
    cfg      = load_config(CONFIG_PATH)['entsoe']
    client   = EntsoePandasClient(api_key=cfg['api_key'])
    country  = cfg.get('country', 'NL')
    start    = cfg.get('default_start')
    neighbors= cfg.get('neighbors', [])

    conn = get_connection(WARP_DB_PATH)

    if table_exists(conn, TABLE_RAW):
        last = pd.read_sql_query(
            f"SELECT MAX(datetime) AS dt FROM {TABLE_RAW}", conn
        )['dt'][0]
        start = last or start

    end = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0).isoformat()

    load_s  = fetch_with_retries(client.query_load, country, start=start, end=end)
    price_s = fetch_with_retries(client.query_day_ahead_prices, country, start=start, end=end)

    flows = {}
    for n in neighbors:
        flows[f'Flow_{n}_to_{country}'] = fetch_with_retries(
            client.query_crossborder_flows,
            country_code_from=n,
            country_code_to=country,
            start=start,
            end=end
        )
        flows[f'Flow_{country}_to_{n}'] = fetch_with_retries(
            client.query_crossborder_flows,
            country_code_from=country,
            country_code_to=n,
            start=start,
            end=end
        )

    df = pd.DataFrame({'load': load_s, 'price': price_s})
    for col, series in flows.items():
        df[col] = series

    df = df.reset_index().rename(columns={'index': 'datetime'})
    df['datetime'] = df['datetime'].astype(str)

    if not table_exists(conn, TABLE_RAW):
        create_table_from_df(conn, df, TABLE_RAW)

    df.to_sql(TABLE_RAW, conn, if_exists='append', index=False)
    remove_duplicates(conn, TABLE_RAW)
    conn.close()

    print(f"Ingested ENTSO-E from {start} to {end} into '{TABLE_RAW}'.")

if __name__ == '__main__':
    ingest_entsoe()