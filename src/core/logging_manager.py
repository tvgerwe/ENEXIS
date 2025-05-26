# ============================================================================
# FILE: src/core/logging_manager.py
# ============================================================================

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager

from config.database_config import DatabaseConfig

class ExperimentLogger:
    """Unified logging system for all experiment results"""
    
    def __init__(self, logs_db_path: Path):
        self.logs_db_path = Path(logs_db_path)
        self.current_experiment_id = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure logs database and tables exist"""
        if not self.logs_db_path.parent.exists():
            self.logs_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table_name, schema in DatabaseConfig.LOGS_DB_SCHEMA.items():
                cursor.execute(schema)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.logs_db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def start_experiment(self, experiment_name: str, config: Dict, notes: Optional[str] = None) -> int:
        """Start a new experiment and return experiment ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (experiment_name, config_json, started_at, notes)
                VALUES (?, ?, ?, ?)
            """, (
                experiment_name,
                json.dumps(config),
                datetime.utcnow().isoformat(),
                notes
            ))
            experiment_id = cursor.lastrowid
            conn.commit()
        
        self.current_experiment_id = experiment_id
        self.logger.info(f"✅ Started experiment '{experiment_name}' with ID {experiment_id}")
        return experiment_id
    
    def finish_experiment(self, experiment_id: Optional[int] = None, status: str = "completed", notes: Optional[str] = None):
        """Mark experiment as finished"""
        exp_id = experiment_id or self.current_experiment_id
        if not exp_id:
            self.logger.warning("No experiment ID provided and no current experiment")
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET completed_at = ?, status = ?, notes = COALESCE(?, notes)
                WHERE id = ?
            """, (
                datetime.utcnow().isoformat(),
                status,
                notes,
                exp_id
            ))
            conn.commit()
        
        self.logger.info(f"✅ Finished experiment {exp_id} with status: {status}")
    
    def log_model_run(self,
                     model_name: str,
                     model_variant: str,
                     train_start: str,
                     train_end: str,
                     forecast_start: str,
                     forecast_end: str,
                     window_id: Optional[int] = None,
                     execution_time: Optional[float] = None,
                     experiment_id: Optional[int] = None) -> int:
        """Log a model run and return run ID"""
        
        exp_id = experiment_id or self.current_experiment_id
        if not exp_id:
            raise ValueError("No experiment ID available. Start an experiment first.")
        
        # Calculate sizes
        train_size = self._calculate_hours_between(train_start, train_end)
        forecast_size = self._calculate_hours_between(forecast_start, forecast_end)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_runs (
                    experiment_id, model_name, model_variant, window_id,
                    train_start, train_end, forecast_start, forecast_end,
                    train_size, forecast_size, execution_time_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, model_name, model_variant, window_id,
                train_start, train_end, forecast_start, forecast_end,
                train_size, forecast_size, execution_time,
                datetime.utcnow().isoformat()
            ))
            run_id = cursor.lastrowid
            conn.commit()
        
        self.logger.info(f"✅ Logged model run for {model_name} ({model_variant}) with ID {run_id}")
        return run_id
    
    def log_model_results(self,
                         model_run_id: int,
                         metrics: Dict[str, float],
                         detailed_metrics: Optional[Dict] = None):
        """Log model performance metrics"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Log main metrics
            for metric_name, metric_value in metrics.items():
                metadata = None
                if detailed_metrics and metric_name in detailed_metrics:
                    metadata = json.dumps(detailed_metrics[metric_name])
                
                cursor.execute("""
                    INSERT INTO model_results (model_run_id, metric_name, metric_value, metric_metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    model_run_id, metric_name, metric_value, metadata,
                    datetime.utcnow().isoformat()
                ))
            
            conn.commit()
        
        self.logger.info(f"✅ Logged {len(metrics)} metrics for model run {model_run_id}")
    
    def log_model_details(self,
                         model_run_id: int,
                         parameters: Optional[Dict] = None,
                         hyperparameters: Optional[Dict] = None,
                         diagnostics: Optional[Dict] = None,
                         convergence_info: Optional[Dict] = None,
                         feature_stats: Optional[Dict] = None,
                         model_summary: Optional[str] = None):
        """Log detailed model information"""
        
        details = {}
        if parameters:
            details['parameters'] = parameters
        if hyperparameters:
            details['hyperparameters'] = hyperparameters
        if diagnostics:
            details['diagnostics'] = diagnostics
        if convergence_info:
            details['convergence_info'] = convergence_info
        if feature_stats:
            details['feature_stats'] = feature_stats
        if model_summary:
            details['model_summary'] = model_summary
        
        if not details:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for detail_type, detail_data in details.items():
                if detail_type == 'model_summary':
                    detail_json = json.dumps({'summary': detail_data})
                else:
                    detail_json = json.dumps(detail_data)
                
                cursor.execute("""
                    INSERT INTO model_details (model_run_id, detail_type, detail_json, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    model_run_id, detail_type, detail_json,
                    datetime.utcnow().isoformat()
                ))
            
            conn.commit()
        
        self.logger.info(f"✅ Logged {len(details)} detail types for model run {model_run_id}")
    
    def update_model_run_status(self, model_run_id: int, status: str, error_message: Optional[str] = None):
        """Update model run status"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE model_runs 
                SET status = ?, error_message = ?
                WHERE id = ?
            """, (status, error_message, model_run_id))
            conn.commit()
    
    def get_experiment_results(self, experiment_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Get experiment results for analysis"""
        exp_id = experiment_id or self.current_experiment_id
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    mr.id as run_id,
                    mr.model_name,
                    mr.model_variant,
                    mr.window_id,
                    mr.train_start,
                    mr.train_end,
                    mr.forecast_start,
                    mr.forecast_end,
                    mr.execution_time_seconds,
                    mr.status,
                    GROUP_CONCAT(res.metric_name || ':' || res.metric_value) as metrics
                FROM model_runs mr
                LEFT JOIN model_results res ON mr.id = res.model_run_id
                WHERE mr.experiment_id = ?
                GROUP BY mr.id
                ORDER BY mr.created_at DESC
                LIMIT ?
            """, (exp_id, limit))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'run_id': row[0],
                    'model_name': row[1],
                    'model_variant': row[2],
                    'window_id': row[3],
                    'train_start': row[4],
                    'train_end': row[5],
                    'forecast_start': row[6],
                    'forecast_end': row[7],
                    'execution_time': row[8],
                    'status': row[9],
                    'metrics': {}
                }
                
                # Parse metrics
                if row[10]:
                    for metric_pair in row[10].split(','):
                        if ':' in metric_pair:
                            name, value = metric_pair.split(':', 1)
                            try:
                                result['metrics'][name] = float(value)
                            except ValueError:
                                result['metrics'][name] = value
                
                results.append(result)
        
        return results
    
    def get_performance_trends(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze performance trends across experiments"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            where_clause = ""
            params = []
            if model_names:
                placeholders = ','.join('?' * len(model_names))
                where_clause = f"WHERE mr.model_name IN ({placeholders})"
                params = model_names
            
            cursor.execute(f"""
                SELECT 
                    mr.model_name,
                    mr.model_variant,
                    mr.window_id,
                    res.metric_name,
                    res.metric_value,
                    mr.created_at
                FROM model_runs mr
                JOIN model_results res ON mr.id = res.model_run_id
                {where_clause}
                ORDER BY mr.created_at DESC
                LIMIT 500
            """, params)
            
            trends = {}
            for row in cursor.fetchall():
                model_key = f"{row[0]}_{row[1]}"
                if model_key not in trends:
                    trends[model_key] = {'metrics': {}, 'windows': set()}
                
                metric_name = row[3]
                metric_value = row[4]
                window_id = row[2]
                
                if metric_name not in trends[model_key]['metrics']:
                    trends[model_key]['metrics'][metric_name] = []
                
                trends[model_key]['metrics'][metric_name].append({
                    'value': metric_value,
                    'window_id': window_id,
                    'timestamp': row[5]
                })
                
                if window_id:
                    trends[model_key]['windows'].add(window_id)
        
        # Calculate trend statistics
        for model_key in trends:
            trends[model_key]['windows'] = list(trends[model_key]['windows'])
            for metric_name in trends[model_key]['metrics']:
                values = [m['value'] for m in trends[model_key]['metrics'][metric_name] if m['value'] is not None]
                if len(values) > 1:
                    trends[model_key]['metrics'][metric_name + '_trend'] = {
                        'first': values[0],
                        'last': values[-1],
                        'change_pct': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0,
                        'count': len(values)
                    }
        
        return trends
    
    def _calculate_hours_between(self, start_str: str, end_str: str) -> Optional[int]:
        """Calculate hours between two ISO datetime strings"""
        try:
            import pandas as pd
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            return int((end - start).total_seconds() / 3600) + 1
        except:
            return None

# ============================================================================
# Legacy compatibility functions (for backward compatibility)
# ============================================================================

def log_rmse_to_sqlite(model_name: str, variant: str, train_start: str, train_end: str,
                      forecast_start: str, forecast_end: str, rmse_overall: float,
                      rmse_per_day: dict, rmse_per_hour: dict, parameters: dict,
                      features_used: list, logs_db_path: Optional[Path] = None):
    """Legacy compatibility wrapper"""
    db_path = logs_db_path or Path("src/data/logs.db")
    logger = ExperimentLogger(db_path)
    
    # Start a legacy experiment if none exists
    if not logger.current_experiment_id:
        logger.start_experiment(f"Legacy_{model_name}_{variant}", {
            'model_name': model_name,
            'variant': variant,
            'parameters': parameters,
            'features_used': features_used
        })
    
    # Log model run
    run_id = logger.log_model_run(
        model_name=model_name,
        model_variant=variant,
        train_start=train_start,
        train_end=train_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end
    )
    
    # Log results
    logger.log_model_results(
        model_run_id=run_id,
        metrics={'rmse_overall': rmse_overall},
        detailed_metrics={
            'rmse_overall': {
                'per_day': rmse_per_day,
                'per_hour': rmse_per_hour
            }
        }
    )
    
    # Log model details
    logger.log_model_details(
        model_run_id=run_id,
        parameters=parameters,
        hyperparameters={'features_used': features_used}
    )

def log_rolling_window_to_sqlite(window_id: int, model_name: str, train_start: str,
                                train_end: str, forecast_start: str, forecast_end: str,
                                rmse: Optional[float], **kwargs):
    """Legacy compatibility wrapper"""
    logs_db_path = kwargs.get('logs_db_path', Path("src/data/logs.db"))
    logger = ExperimentLogger(logs_db_path)
    
    # Start a legacy experiment if none exists
    if not logger.current_experiment_id:
        logger.start_experiment(f"Legacy_RollingWindow_{model_name}", {
            'model_name': model_name,
            'window_id': window_id
        })
    
    # Log model run
    run_id = logger.log_model_run(
        model_name=model_name,
        model_variant=f"window_{window_id}",
        train_start=train_start,
        train_end=train_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        window_id=window_id,
        execution_time=kwargs.get('execution_time')
    )
    
    # Log results
    metrics = {}
    if rmse is not None:
        metrics['rmse'] = rmse
    if 'mae' in kwargs and kwargs['mae'] is not None:
        metrics['mae'] = kwargs['mae']
    if 'mape' in kwargs and kwargs['mape'] is not None:
        metrics['mape'] = kwargs['mape']
    
    if metrics:
        logger.log_model_results(model_run_id=run_id, metrics=metrics)
    
    # Log detailed information
    detail_keys = ['model_parameters', 'hyperparameters', 'model_diagnostics', 
                   'convergence_info', 'feature_stats', 'model_summary']
    details = {k: kwargs[k] for k in detail_keys if k in kwargs and kwargs[k] is not None}
    
    if details:
        logger.log_model_details(model_run_id=run_id, **details)