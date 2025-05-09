#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd

def clean_ned_obs(df):
    df = df[['capacity','volume','percentage','validfrom']].copy()
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').astype('Int64')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')
    df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce').astype(float)
    df['validfrom'] = pd.to_datetime(df['validfrom'], utc=True)
    return df.rename(columns=lambda c: f'ned.{c}')

def load_transform_ned_obs(df, db_path, table_name='transform_ned_obs'):
    folder = os.path.dirname(db_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    conn = sqlite3.connect(db_path)
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS {table_name} ('
        '"ned.capacity" INTEGER,'
        '"ned.volume" INTEGER,'
        '"ned.percentage" REAL,'
        '"ned.validfrom" TEXT'
        ')'
    )
    conn.commit()
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.execute(
        f'DELETE FROM {table_name} WHERE rowid NOT IN ('
        f'SELECT MIN(rowid) FROM {table_name} GROUP BY '
        '"ned.capacity","ned.volume","ned.percentage","ned.validfrom"'
        ')'
    )
    conn.commit()
    conn.close()

def transform_ned_pipeline(db_path=None, raw_table='raw_ned_obs', transform_table='transform_ned_obs'):
    if db_path is None:
        here = os.path.dirname(__file__)
        db_path = os.path.abspath(os.path.join(here, '..', 'data', 'WARP.db'))
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {raw_table}", conn)
    conn.close()
    df2 = clean_ned_obs(df)
    load_transform_ned_obs(df2, db_path, transform_table)

if __name__ == '__main__':
    transform_ned_pipeline()