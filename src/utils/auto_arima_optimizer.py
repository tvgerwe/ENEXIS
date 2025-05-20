# src/utils/auto_arima_optimizer.py

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pmdarima import auto_arima

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - auto_arima - %(levelname)s - %(message)s")
logger = logging.getLogger("auto_arima_optimizer")

# Paden
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WARP_DB = PROJECT_ROOT / "src" / "data" / "WARP.db"
LOG_DB = PROJECT_ROOT / "src" / "data" / "logs.db"
DATA_TABLE = "master_warp"
LOG_TABLE = "arima_configs"

def ensure_log_table():
    """Zorgt dat de loggingtabel bestaat in logs.db."""
    conn = sqlite3.connect(LOG_DB)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            order_params TEXT NOT NULL,
            seasonal_order_params TEXT NOT NULL,
            aic REAL,
            bic REAL,
            lambda REAL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def find_existing_config(train_start: str, train_end: str) -> Optional[Tuple]:
    """Zoekt eerder gelogde auto-arima configuratie op dezelfde datarange."""
    conn = sqlite3.connect(LOG_DB)
    cursor = conn.cursor()
    query = f"""
        SELECT order_params, seasonal_order_params, lambda
        FROM {LOG_TABLE}
        WHERE model = 'SARIMA'
        AND train_start = ?
        AND train_end = ?
        AND status = 'success'
        ORDER BY id DESC
        LIMIT 1
    """
    cursor.execute(query, (train_start, train_end))
    row = cursor.fetchone()
    conn.close()
    if row:
        logger.info("⚡ Eerdere auto_arima-configuratie hergebruikt.")
        return eval(row[0]), eval(row[1]), row[2]
    return None

def get_best_sarima_params_from_db(train_days: int = 28) -> Tuple[Tuple, Tuple, Optional[float]]:
    """
    Voert auto_arima uit op de laatste `train_days` uit master_warp.
    Hergebruikt bestaande settings indien al gelogd.
    """

    ensure_log_table()

    if not WARP_DB.exists():
        raise FileNotFoundError(f"Database niet gevonden: {WARP_DB}")

    conn = sqlite3.connect(WARP_DB)
    df = pd.read_sql_query(f"SELECT target_datetime, Price FROM {DATA_TABLE}", conn, parse_dates=["target_datetime"])
    conn.close()

    df.set_index("target_datetime", inplace=True)
    df = df.asfreq("H")

    train_end = df.index.max()
    train_start = train_end - pd.Timedelta(days=train_days)

    df_train = df.loc[train_start:train_end]["Price"].dropna()

    # Format voor logging
    train_start_str = train_start.strftime("%Y-%m-%d %H:%M:%S")
    train_end_str = train_end.strftime("%Y-%m-%d %H:%M:%S")

    # Zoek bestaande configuratie
    existing = find_existing_config(train_start_str, train_end_str)
    if existing:
        return existing

    try:
        logger.info("🚀 Start auto_arima optimalisatie...")
        model = auto_arima(
            df_train,
            seasonal=True,
            m=24,
            stepwise=True,
            d=1,
            D=1,
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            start_p=1,
            start_q=1,
            start_P=1,
            start_Q=1,
            trend='t',
            lambda_=None,
            suppress_warnings=True,
            error_action="warn",
            max_order=10,
            information_criterion="aic"
        )

        order = model.order
        seasonal_order = model.seasonal_order
        lambda_val = getattr(model, 'lambda_', None)
        aic = model.aic()
        bic = model.bic()

        logger.info(f"✅ Beste configuratie: order={order}, seasonal={seasonal_order}")

        # Log naar logs.db
        conn = sqlite3.connect(LOG_DB)
        conn.execute(f"""
            INSERT INTO {LOG_TABLE} (
                model, train_start, train_end, order_params,
                seasonal_order_params, aic, bic, lambda, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "SARIMA",
            train_start_str,
            train_end_str,
            str(order),
            str(seasonal_order),
            aic,
            bic,
            lambda_val,
            "success",
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

        return order, seasonal_order, lambda_val

    except Exception as e:
        logger.error(f"❌ Auto-ARIMA gefaald: {e}")
        # Zoek fallback uit eerdere runs
        fallback = find_existing_config(train_start_str, train_end_str)
        if fallback:
            logger.warning("🛟 Fallback naar vorige gelogde configuratie.")
            return fallback
        else:
            logger.warning("🛟 Fallback naar standaardwaarden.")
            return (2,1,1), (1,1,1,24), None
