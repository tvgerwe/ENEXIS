# src/config/database_config.py
from pathlib import Path
from typing import Dict

class DatabaseConfig:
    """Database configuration and schema management"""
    
    WARP_DB_SCHEMA = {
        'master_warp': {
            'primary_key': 'target_datetime',
            'required_columns': ['target_datetime', 'Price'],
            'index_columns': ['target_datetime']
        },
        'master_predictions': {
            'primary_key': 'target_datetime,run_date',
            'required_columns': ['target_datetime', 'run_date'],
            'index_columns': ['target_datetime', 'run_date']
        },
        'training_set': {
            'primary_key': 'target_datetime',
            'required_columns': ['target_datetime'],
            'index_columns': ['target_datetime']
        }
    }
    
    LOGS_DB_SCHEMA = {
        'experiments': """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'running',
                notes TEXT
            )
        """,
        'model_runs': """
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_variant TEXT NOT NULL,
                window_id INTEGER,
                train_start TEXT NOT NULL,
                train_end TEXT NOT NULL,
                forecast_start TEXT NOT NULL,
                forecast_end TEXT NOT NULL,
                train_size INTEGER,
                forecast_size INTEGER,
                execution_time_seconds REAL,
                status TEXT DEFAULT 'running',
                error_message TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """,
        'model_results': """
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_run_id) REFERENCES model_runs(id)
            )
        """,
        'model_details': """
            CREATE TABLE IF NOT EXISTS model_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_run_id INTEGER NOT NULL,
                detail_type TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_run_id) REFERENCES model_runs(id)
            )
        """
    }

def get_database_schemas() -> Dict[str, Dict]:
    """Get all database schemas"""
    return {
        'warp_db': DatabaseConfig.WARP_DB_SCHEMA,
        'logs_db': DatabaseConfig.LOGS_DB_SCHEMA
    }