# src/utils/logger.py

import os
import sqlite3
from datetime import datetime

# Dynamisch pad (altijd relatief vanaf deze file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DB_PATH = os.path.join(BASE_DIR, "src", "data", "logs.db")
TABLE_NAME = "pipeline_logs"

def init_logger():
    os.makedirs(os.path.dirname(LOG_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(LOG_DB_PATH)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            module TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_info(message: str, module: str = ""):
    _log("INFO", message, module)

def log_error(message: str, module: str = ""):
    _log("ERROR", message, module)

def _log(level: str, message: str, module: str):
    conn = sqlite3.connect(LOG_DB_PATH)
    timestamp = datetime.utcnow().isoformat()
    conn.execute(f"""
        INSERT INTO {TABLE_NAME} (timestamp, level, message, module)
        VALUES (?, ?, ?, ?)
    """, (timestamp, level, message, module))
    conn.commit()
    conn.close()
