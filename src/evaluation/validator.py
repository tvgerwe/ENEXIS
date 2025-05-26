import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from core.data_manager import DataManager, DataSplit
from models.factory import ModelFactory, ModelResult
from evaluation.metrics import MetricsCalculator
from core.logging_manager import ExperimentLogger

class RollingWindowValidator:
    """Rolling window cross-validation for time series models"""
    
    def __init__(self, 
                 data_manager: DataManager,
                 model_factory: ModelFactory,
                 logger: ExperimentLogger,
                 metrics_calculator: Optional[MetricsCalculator] = None):
        self.data_manager = data_manager
        self.model_factory = model_factory
        self.logger_manager = logger
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, 
                 n_windows: int = 3,
                 parallel: bool = False,
                 max_workers: Optional[int] = None) -> pd.DataFrame:
        """Perform rolling window validation"""
        
        self.logger.info(f"🔄 Starting rolling window validation with {n_windows} windows")
        
        # Create rolling data splits
        data_splits = self.data_manager.create_rolling_splits(n_windows)
        
        if not data_splits:
            self.logger.error("❌ No valid data splits created")
            return pd.DataFrame()
        
        self.logger.info(f"✅ Created {len(data_splits)} data splits")
        
        # Run validation
        if parallel and len(data_splits) > 1:
            results = self._validate_parallel(data_splits, max_workers)
        else:
            results = self._validate_sequential(data_splits)
        
        # Convert results to DataFrame
        results_df = self._results_to_dataframe(results)
        
        self.logger.info(f"✅ Rolling window validation complete")
        return results_df
    
    def _validate_sequential(self, data_splits: List[DataSplit]) -> List[Dict]:
        """Run validation sequentially"""
        results = []
        
        for i, data_split in enumerate(data_splits, 1):
            self.logger.info(f"📊 Processing window {i}/{len(data_splits)}")
            
            window_results = self._validate_single_window(i, data_split)
            results.extend(window_results)
        
        return results
    
    def _validate_parallel(self, data_splits: List[DataSplit], max_workers: Optional[int] = None) -> List[Dict]:
        """Run validation in parallel"""
        max_workers = max_workers or min(len(data_splits), mp.cpu_count())
        results = []
        
        self.logger.info(f"🚀 Running validation in parallel with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_window = {}
            for i, data_split in enumerate(data_splits, 1):
                future = executor.submit(self._validate_single_window, i, data_split)
                future_to_window[future] = i
            
            # Collect results
            for future in as_completed(future_to_window):
                window_id = future_to_window[future]
                try:
                    window_results = future.result(timeout=600)  # 10 minute timeout
                    results.extend(window_results)
                    self.logger.info(f"✅ Window {window_id} completed")
                except Exception as e:
                    self.logger.error(f"❌ Window {window_id} failed: {e}")
        
        return results
    
    def _validate_single_window(self, window_id: int, data_split: DataSplit) -> List[Dict]:
        """Validate all models on a single window"""
        window_results = []
        
        # Get all model results for this window
        model_results = self.model_factory.run_all_models(data_split)
        
        for model_name, model_result in model_results.items():
            try:
                # Log model run
                run_id = self.logger_manager.log_model_run(
                    model_name=model_result.model_name,
                    model_variant=model_result.model_variant,
                    train_start=data_split.train_start.isoformat(),
                    train_end=data_split.train_end.isoformat(),
                    forecast_start=data_split.forecast_start.isoformat(),
                    forecast_end=data_split.forecast_end.isoformat(),
                    window_id=window_id,
                    execution_time=model_result.execution_time
                )
                
                if model_result.success:
                    # Calculate metrics
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        data_split.y_test, 
                        model_result.predictions
                    )
                    
                    detailed_rmse = self.metrics_calculator.calculate_detailed_rmse(
                        data_split.y_test,
                        model_result.predictions
                    )
                    
                    statistical_metrics = self.metrics_calculator.calculate_statistical_metrics(
                        data_split.y_test,
                        model_result.predictions
                    )
                    
                    # Calculate data statistics
                    train_stats = self.metrics_calculator.calculate_data_statistics(data_split.y_train, "train")
                    actual_stats = self.metrics_calculator.calculate_data_statistics(data_split.y_test, "actual")
                    pred_stats = self.metrics_calculator.calculate_data_statistics(model_result.predictions, "pred")
                    
                    # Log results
                    self.logger_manager.log_model_results(
                        model_run_id=run_id,
                        metrics=metrics,
                        detailed_metrics={'rmse_detailed': detailed_rmse}
                    )
                    
                    # Log model details
                    self.logger_manager.log_model_details(
                        model_run_id=run_id,
                        parameters=model_result.parameters,
                        hyperparameters=model_result.hyperparameters,
                        diagnostics=model_result.diagnostics,
                        convergence_info=model_result.convergence_info,
                        model_summary=model_result.model_summary
                    )
                    
                    # Update status
                    self.logger_manager.update_model_run_status(run_id, "completed")
                    
                    # Store result for DataFrame
                    window_results.append({
                        'window_id': window_id,
                        'model_name': model_name,
                        'model_variant': model_result.model_variant,
                        'train_start': data_split.train_start,
                        'train_end': data_split.train_end,
                        'forecast_start': data_split.forecast_start,
                        'forecast_end': data_split.forecast_end,
                        'execution_time': model_result.execution_time,
                        'status': 'completed',
                        **metrics,
                        **statistical_metrics,
                        **train_stats,
                        **actual_stats,
                        **pred_stats
                    })
                    
                    self.logger.info(f"  ✅ {model_name}: RMSE = {metrics.get('rmse', 'N/A'):.6f}")
                    
                else:
                    # Handle failed model
                    self.logger_manager.update_model_run_status(
                        run_id, "failed", 
                        model_result.error_message
                    )
                    
                    window_results.append({
                        'window_id': window_id,
                        'model_name': model_name,
                        'model_variant': model_result.model_variant,
                        'train_start': data_split.train_start,
                        'train_end': data_split.train_end,
                        'forecast_start': data_split.forecast_start,
                        'forecast_end': data_split.forecast_end,
                        'execution_time': model_result.execution_time,
                        'status': 'failed',
                        'error_message': model_result.error_message,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'mape': np.nan
                    })
                    
                    self.logger.error(f"  ❌ {model_name}: {model_result.error_message}")
            
            except Exception as e:
                self.logger.error(f"  ❌ Error processing {model_name} in window {window_id}: {e}")
                window_results.append({
                    'window_id': window_id,
                    'model_name': model_name,
                    'model_variant': 'unknown',
                    'execution_time': 0,
                    'status': 'error',
                    'error_message': str(e),
                    'rmse': np.nan,
                    'mae': np.nan,
                    'mape': np.nan
                })
        
        return window_results
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results list to DataFrame"""
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Ensure consistent columns
        required_columns = [
            'window_id', 'model_name', 'model_variant', 'train_start', 'train_end',
            'forecast_start', 'forecast_end', 'execution_time', 'status',
            'rmse', 'mae', 'mape'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Sort by window and model
        df = df.sort_values(['window_id', 'model_name']).reset_index(drop=True)
        
        return df
    
    def analyze_performance_trends(self, results_df: pd.DataFrame) -> Dict:
        """Analyze performance trends across rolling windows"""
        if results_df.empty:
            return {}
        
        analysis = {
            'total_windows': results_df['window_id'].nunique(),
            'models_tested': results_df['model_name'].unique().tolist(),
            'success_rate': {},
            'performance_trends': {},
            'degradation_analysis': {}
        }
        
        # Success rate by model
        for model in analysis['models_tested']:
            model_results = results_df[results_df['model_name'] == model]
            success_count = (model_results['status'] == 'completed').sum()
            total_count = len(model_results)
            analysis['success_rate'][model] = {
                'success_count': success_count,
                'total_count': total_count,
                'success_rate': (success_count / total_count) * 100 if total_count > 0 else 0
            }
        
        # Performance trends
        for model in analysis['models_tested']:
            model_results = results_df[
                (results_df['model_name'] == model) & 
                (results_df['status'] == 'completed')
            ].sort_values('window_id')
            
            if len(model_results) > 1:
                rmse_values = model_results['rmse'].dropna()
                if len(rmse_values) >= 2:
                    first_rmse = rmse_values.iloc[0]
                    last_rmse = rmse_values.iloc[-1]
                    
                    degradation_pct = ((last_rmse - first_rmse) / first_rmse) * 100 if first_rmse != 0 else 0
                    
                    analysis['performance_trends'][model] = {
                        'first_window_rmse': first_rmse,
                        'last_window_rmse': last_rmse,
                        'degradation_percent': degradation_pct,
                        'trend': 'SEVERE' if degradation_pct > 100 else 
                                'SIGNIFICANT' if degradation_pct > 50 else
                                'MODERATE' if degradation_pct > 20 else
                                'STABLE' if abs(degradation_pct) <= 20 else 'IMPROVING',
                        'rmse_values': rmse_values.tolist(),
                        'windows_completed': len(rmse_values)
                    }
        
        return analysis