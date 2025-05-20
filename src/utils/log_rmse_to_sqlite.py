 #!/usr/bin/env python3

import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Logging configuratie
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Pad naar de database
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "logs.db"
TABLE_NAME = "model_rmse_logs"

def ensure_directory_exists(path: Path):
    """Zorgt dat de directory voor de database bestaat."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def validate_rmse_structure(per_day: dict, per_hour: dict):
    """Valideert structuur van RMSE per dag en per uur."""
    if not all(str(k) in per_day for k in range(1, 8)):
        raise ValueError("❌ rmse_per_day moet keys bevatten van '1' t/m '7'")
    if not all(str(k) in per_hour for k in range(168)):
        raise ValueError("❌ rmse_per_hour moet keys bevatten van '0' t/m '167'")

def log_rmse_to_sqlite(
    model_name: str,
    variant: str,
    train_start: str,
    train_end: str,
    forecast_start: str,
    forecast_end: str,
    rmse_overall: float,
    rmse_per_day: dict,
    rmse_per_hour: dict,
    parameters: dict,
    features_used: list
):
    """
    Logt een RMSE-resultaat naar de centrale logs.db
    Spec vereist consistent gebruik van str keys voor dicts (i.p.v. ints).
    """

    ensure_directory_exists(DB_PATH)
    validate_rmse_structure(rmse_per_day, rmse_per_hour)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            variant TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            forecast_start TEXT NOT NULL,
            forecast_end TEXT NOT NULL,
            forecast_horizon INTEGER NOT NULL,
            rmse_json TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            features_used_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    rmse_json = json.dumps({
        "overall": rmse_overall,
        "per_day": rmse_per_day,
        "per_hour": rmse_per_hour
    })

    parameters_json = json.dumps(parameters)
    features_used_json = json.dumps(features_used)

    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (
            model_name, variant, train_start, train_end,
            forecast_start, forecast_end, forecast_horizon,
            rmse_json, parameters_json, features_used_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        variant,
        train_start,
        train_end,
        forecast_start,
        forecast_end,
        168,
        rmse_json,
        parameters_json,
        features_used_json,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()
    logger.info(f"✅ RMSE-log succesvol opgeslagen voor model: {model_name} ({variant})")
