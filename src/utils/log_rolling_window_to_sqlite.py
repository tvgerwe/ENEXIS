#!/usr/bin/env python3

import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Logging configuratie
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Pad naar de database
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "logs.db"
TABLE_NAME = "rolling_window_logs"

def ensure_directory_exists(path: Path):
    """Zorgt dat de directory voor de database bestaat."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def log_rolling_window_to_sqlite(
    window_id: int,
    model_name: str, 
    train_start: str,
    train_end: str,
    forecast_start: str,
    forecast_end: str,
    rmse: Optional[float],
    mae: Optional[float] = None,
    mape: Optional[float] = None,
    train_stats: Optional[Dict] = None,
    forecast_stats: Optional[Dict] = None, 
    actual_stats: Optional[Dict] = None,
    model_diagnostics: Optional[Dict] = None,
    feature_stats: Optional[Dict] = None,
    performance_metrics: Optional[Dict] = None,
    model_parameters: Optional[Dict] = None,
    hyperparameters: Optional[Dict] = None,
    model_summary: Optional[str] = None,
    convergence_info: Optional[Dict] = None,
    execution_time: Optional[float] = None,
    notes: Optional[str] = None
):
    """
    Logt rolling window validatie resultaten naar de centrale logs.db
    
    Args:
        window_id: Rolling window nummer (1, 2, 3, ...)
        model_name: Model naam (naive, sarimax_no_exog, sarimax_with_exog)
        train_start/end: Training period
        forecast_start/end: Forecast period  
        rmse: Root Mean Square Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        train_stats: {'mean': float, 'std': float, 'min': float, 'max': float}
        forecast_stats: Stats van predictions
        actual_stats: Stats van actual values
        model_diagnostics: {'aic': float, 'bic': float, 'converged': bool, 'warnings': str}
        feature_stats: Feature drift en statistieken
        performance_metrics: Extra metrics zoals seasonal_strength, data_drift_score
        model_parameters: Fitted model parameters (coefficients, etc.)
        hyperparameters: Model hyperparameters (order, seasonal_order, etc.)
        model_summary: String representation of model summary
        convergence_info: Convergence diagnostics and warnings
        execution_time: Tijd in seconden
        notes: Extra opmerkingen
    """
    
    ensure_directory_exists(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            forecast_start TEXT NOT NULL,
            forecast_end TEXT NOT NULL,
            train_size INTEGER,
            forecast_size INTEGER,
            rmse REAL,
            mae REAL,
            mape REAL,
            train_stats_json TEXT,
            forecast_stats_json TEXT,
            actual_stats_json TEXT,
            model_diagnostics_json TEXT,
            feature_stats_json TEXT,
            performance_metrics_json TEXT,
            model_parameters_json TEXT,
            hyperparameters_json TEXT,
            model_summary TEXT,
            convergence_info_json TEXT,
            execution_time_seconds REAL,
            notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    
    # Calculate train/forecast sizes from date ranges
    train_size = None
    forecast_size = None
    try:
        from pandas import to_datetime, Timedelta
        train_start_dt = to_datetime(train_start)
        train_end_dt = to_datetime(train_end)
        forecast_start_dt = to_datetime(forecast_start)
        forecast_end_dt = to_datetime(forecast_end)
        
        train_size = int((train_end_dt - train_start_dt).total_seconds() / 3600) + 1
        forecast_size = int((forecast_end_dt - forecast_start_dt).total_seconds() / 3600) + 1
    except:
        pass
    
    # Convert dictionaries to JSON strings
    train_stats_json = json.dumps(train_stats) if train_stats else None
    forecast_stats_json = json.dumps(forecast_stats) if forecast_stats else None
    actual_stats_json = json.dumps(actual_stats) if actual_stats else None
    model_diagnostics_json = json.dumps(model_diagnostics) if model_diagnostics else None
    feature_stats_json = json.dumps(feature_stats) if feature_stats else None
    performance_metrics_json = json.dumps(performance_metrics) if performance_metrics else None
    model_parameters_json = json.dumps(model_parameters) if model_parameters else None
    hyperparameters_json = json.dumps(hyperparameters) if hyperparameters else None
    convergence_info_json = json.dumps(convergence_info) if convergence_info else None
    
    # Insert record
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (
            window_id, model_name, train_start, train_end, forecast_start, forecast_end,
            train_size, forecast_size, rmse, mae, mape,
            train_stats_json, forecast_stats_json, actual_stats_json,
            model_diagnostics_json, feature_stats_json, performance_metrics_json,
            model_parameters_json, hyperparameters_json, model_summary, convergence_info_json,
            execution_time_seconds, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        window_id, model_name, train_start, train_end, forecast_start, forecast_end,
        train_size, forecast_size, rmse, mae, mape,
        train_stats_json, forecast_stats_json, actual_stats_json,
        model_diagnostics_json, feature_stats_json, performance_metrics_json,
        model_parameters_json, hyperparameters_json, model_summary, convergence_info_json,
        execution_time_seconds, notes, datetime.utcnow().isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Rolling window log succesvol opgeslagen voor window {window_id}, model: {model_name}")

def get_rolling_window_analysis(limit: int = 50) -> list:
    """
    Haalt recente rolling window resultaten op voor analyse
    
    Returns:
        List van dictionaries met rolling window data
    """
    ensure_directory_exists(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT 
            window_id, model_name, rmse, mae, mape,
            train_start, train_end, forecast_start, forecast_end,
            train_stats_json, actual_stats_json, performance_metrics_json,
            execution_time_seconds, notes, created_at
        FROM {TABLE_NAME}
        ORDER BY created_at DESC, window_id ASC
        LIMIT ?
    """, (limit,))
    
    results = []
    for row in cursor.fetchall():
        result = {
            'window_id': row[0],
            'model_name': row[1], 
            'rmse': row[2],
            'mae': row[3],
            'mape': row[4],
            'train_start': row[5],
            'train_end': row[6],
            'forecast_start': row[7],
            'forecast_end': row[8],
            'train_stats': json.loads(row[9]) if row[9] else None,
            'actual_stats': json.loads(row[10]) if row[10] else None,
            'performance_metrics': json.loads(row[11]) if row[11] else None,
            'execution_time': row[12],
            'notes': row[13],
            'created_at': row[14]
        }
        results.append(result)
    
    conn.close()
    return results

def analyze_performance_degradation() -> Dict[str, Any]:
    """
    Analyseert performance degradatie over rolling windows
    
    Returns:
        Dictionary met degradatie analyse
    """
    results = get_rolling_window_analysis()
    
    # Group by model and analyze trends
    model_trends = {}
    for result in results:
        model_name = result['model_name']
        if model_name not in model_trends:
            model_trends[model_name] = []
        model_trends[model_name].append(result)
    
    analysis = {
        'total_windows': len(set(r['window_id'] for r in results)),
        'models_analyzed': list(model_trends.keys()),
        'degradation_summary': {}
    }
    
    for model_name, model_results in model_trends.items():
        if len(model_results) < 2:
            continue
            
        # Sort by window_id
        model_results.sort(key=lambda x: x['window_id'])
        
        first_rmse = model_results[0]['rmse']
        last_rmse = model_results[-1]['rmse']
        
        if first_rmse and last_rmse:
            degradation_pct = ((last_rmse - first_rmse) / first_rmse) * 100
            
            analysis['degradation_summary'][model_name] = {
                'first_window_rmse': first_rmse,
                'last_window_rmse': last_rmse,
                'degradation_percent': degradation_pct,
                'trend': 'SEVERE' if degradation_pct > 100 else 
                        'SIGNIFICANT' if degradation_pct > 50 else
                        'MODERATE' if degradation_pct > 20 else
                        'STABLE' if abs(degradation_pct) <= 20 else 'IMPROVING'
            }
    
    return analysis