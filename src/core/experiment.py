# ============================================================================
# FILE: src/core/experiment.py
# ============================================================================

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from config.experiment_config import ExperimentConfig
from core.data_manager import DataManager, DataSplit
from models.factory import ModelFactory, ModelResult
from evaluation.metrics import MetricsCalculator
from evaluation.validator import RollingWindowValidator
from core.logging_manager import ExperimentLogger

class TimeSeriesExperiment:
    """Main experiment orchestrator for time series forecasting"""
    
    def __init__(self, 
                 config: ExperimentConfig,
                 data_manager: DataManager,
                 logger: ExperimentLogger):
        self.config = config
        self.data_manager = data_manager
        self.logger_manager = logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.model_factory = ModelFactory(config.model_configs)
        self.metrics_calculator = MetricsCalculator()
        self.validator = RollingWindowValidator(
            data_manager, self.model_factory, logger, self.metrics_calculator
        )
        
        # Experiment state
        self.experiment_id = None
        self.single_run_results = {}
        self.rolling_results = None
        
    def run_single_experiment(self, 
                            data_split: Optional[DataSplit] = None,
                            save_results: bool = True) -> Dict[str, ModelResult]:
        """Run single experiment with all models"""
        
        self.logger.info("🚀 Starting single experiment run")
        
        # Create data split if not provided
        if data_split is None:
            data_split = self.data_manager.create_splits()
        
        # Log data split info
        split_info = data_split.get_info()
        self.logger.info(f"📊 Data split: {split_info['train_samples']} train, {split_info['test_samples']} test samples")
        
        # Run all models
        start_time = time.time()
        model_results = self.model_factory.run_all_models(
            data_split, 
            parallel=self.config.parallel_execution
        )
        total_time = time.time() - start_time
        
        # Log results if requested
        if save_results and self.experiment_id:
            self._log_single_experiment_results(model_results, data_split)
        
        # Store results
        self.single_run_results = model_results
        
        # Summary
        successful_models = [name for name, result in model_results.items() if result.success]
        failed_models = [name for name, result in model_results.items() if not result.success]
        
        self.logger.info(f"✅ Single experiment complete in {total_time:.2f}s")
        self.logger.info(f"   Successful: {len(successful_models)} models")
        self.logger.info(f"   Failed: {len(failed_models)} models")
        
        if failed_models:
            self.logger.warning(f"   Failed models: {failed_models}")
        
        return model_results
    
    def run_rolling_validation(self, 
                             n_windows: Optional[int] = None,
                             parallel: bool = None,
                             save_results: bool = True) -> pd.DataFrame:
        """Run rolling window validation"""
        
        n_windows = n_windows or self.config.rolling_windows
        parallel = parallel if parallel is not None else self.config.parallel_execution
        
        self.logger.info(f"🔄 Starting rolling window validation ({n_windows} windows)")
        
        # Run validation
        start_time = time.time()
        rolling_results = self.validator.validate(
            n_windows=n_windows,
            parallel=parallel,
            max_workers=self.config.max_workers
        )
        total_time = time.time() - start_time
        
        # Store results
        self.rolling_results = rolling_results
        
        # Analyze trends
        if not rolling_results.empty:
            trends = self.validator.analyze_performance_trends(rolling_results)
            
            self.logger.info(f"✅ Rolling validation complete in {total_time:.2f}s")
            self.logger.info(f"   Total windows: {trends.get('total_windows', 0)}")
            self.logger.info(f"   Models tested: {len(trends.get('models_tested', []))}")
            
            # Log success rates
            for model, success_info in trends.get('success_rate', {}).items():
                rate = success_info['success_rate']
                self.logger.info(f"   {model}: {rate:.1f}% success rate")
        
        return rolling_results
    
    def run_full_experiment(self, 
                          experiment_name: Optional[str] = None,
                          include_rolling: bool = True) -> Dict[str, Any]:
        """Run complete experiment including single run and rolling validation"""
        
        experiment_name = experiment_name or f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start experiment logging
        self.experiment_id = self.logger_manager.start_experiment(
            experiment_name=experiment_name,
            config=self.config.to_dict(),
            notes=f"Full experiment with {len(self.config.model_configs)} models"
        )
        
        self.logger.info(f"🎯 Starting full experiment: {experiment_name}")
        
        results = {
            'experiment_id': self.experiment_id,
            'experiment_name': experiment_name,
            'config': self.config.to_dict(),
            'single_run_results': {},
            'rolling_results': None,
            'summary': {}
        }
        
        try:
            # Run single experiment
            single_results = self.run_single_experiment(save_results=True)
            results['single_run_results'] = self._serialize_model_results(single_results)
            
            # Run rolling validation if requested
            if include_rolling:
                rolling_df = self.run_rolling_validation(save_results=True)
                results['rolling_results'] = rolling_df.to_dict('records') if not rolling_df.empty else []
            
            # Generate summary
            results['summary'] = self._generate_experiment_summary(single_results, rolling_df if include_rolling else None)
            
            # Mark experiment as completed
            self.logger_manager.finish_experiment(
                experiment_id=self.experiment_id,
                status="completed",
                notes="Full experiment completed successfully"
            )
            
            self.logger.info(f"🎉 Full experiment '{experiment_name}' completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {e}", exc_info=True)
            
            # Mark experiment as failed
            self.logger_manager.finish_experiment(
                experiment_id=self.experiment_id,
                status="failed",
                notes=f"Experiment failed: {str(e)}"
            )
            
            results['error'] = str(e)
            results['status'] = 'failed'
        
        return results
    
    def _log_single_experiment_results(self, 
                                     model_results: Dict[str, ModelResult], 
                                     data_split: DataSplit):
        """Log single experiment results to database"""
        
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
                    
                    self.logger_manager.update_model_run_status(run_id, "completed")
                    
                else:
                    # Handle failed model
                    self.logger_manager.update_model_run_status(
                        run_id, "failed", 
                        model_result.error_message
                    )
                
            except Exception as e:
                self.logger.error(f"Error logging results for {model_name}: {e}")
    
    def _serialize_model_results(self, model_results: Dict[str, ModelResult]) -> Dict:
        """Convert ModelResult objects to serializable format"""
        serialized = {}
        
        for model_name, result in model_results.items():
            serialized[model_name] = {
                'model_name': result.model_name,
                'model_variant': result.model_variant,
                'execution_time': result.execution_time,
                'success': result.success,
                'error_message': result.error_message,
                'parameters': result.parameters,
                'hyperparameters': result.hyperparameters,
                'diagnostics': result.diagnostics,
                'convergence_info': result.convergence_info,
                'predictions_summary': {
                    'count': len(result.predictions) if result.predictions is not None else 0,
                    'mean': float(result.predictions.mean()) if result.predictions is not None else None,
                    'std': float(result.predictions.std()) if result.predictions is not None else None,
                    'min': float(result.predictions.min()) if result.predictions is not None else None,
                    'max': float(result.predictions.max()) if result.predictions is not None else None
                }
            }
        
        return serialized
    
    def _generate_experiment_summary(self, 
                                   single_results: Dict[str, ModelResult],
                                   rolling_results: Optional[pd.DataFrame] = None) -> Dict:
        """Generate experiment summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'single_run_summary': {},
            'rolling_validation_summary': {},
            'overall_best_model': None,
            'recommendations': []
        }
        
        # Single run summary
        successful_single = {name: result for name, result in single_results.items() if result.success}
        
        if successful_single:
            # Calculate metrics for comparison
            data_split = self.data_manager.create_splits()
            comparison_metrics = {}
            
            for model_name, result in successful_single.items():
                metrics = self.metrics_calculator.calculate_all_metrics(
                    data_split.y_test, 
                    result.predictions
                )
                comparison_metrics[model_name] = metrics
            
            # Find best model by RMSE
            valid_rmse = {
                model: metrics['rmse'] 
                for model, metrics in comparison_metrics.items()
                if not np.isnan(metrics.get('rmse', np.nan))
            }
            
            if valid_rmse:
                best_single_model = min(valid_rmse.items(), key=lambda x: x[1])
                summary['single_run_summary'] = {
                    'total_models': len(single_results),
                    'successful_models': len(successful_single),
                    'failed_models': len(single_results) - len(successful_single),
                    'best_model': best_single_model[0],
                    'best_rmse': best_single_model[1],
                    'metrics_comparison': comparison_metrics
                }
                summary['overall_best_model'] = best_single_model[0]
        
        # Rolling validation summary
        if rolling_results is not None and not rolling_results.empty:
            trends = self.validator.analyze_performance_trends(rolling_results)
            summary['rolling_validation_summary'] = trends
            
            # Overall best model considering both single and rolling
            rolling_avg_rmse = {}
            for model in trends.get('models_tested', []):
                model_results = rolling_results[
                    (rolling_results['model_name'] == model) & 
                    (rolling_results['status'] == 'completed')
                ]
                if len(model_results) > 0:
                    avg_rmse = model_results['rmse'].mean()
                    if not np.isnan(avg_rmse):
                        rolling_avg_rmse[model] = avg_rmse
            
            if rolling_avg_rmse:
                best_rolling_model = min(rolling_avg_rmse.items(), key=lambda x: x[1])
                summary['rolling_validation_summary']['best_model'] = best_rolling_model[0]
                summary['rolling_validation_summary']['best_avg_rmse'] = best_rolling_model[1]
                
                # Update overall best if rolling results are available
                summary['overall_best_model'] = best_rolling_model[0]
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate recommendations based on experiment results"""
        recommendations = []
        
        # Single run recommendations
        single_summary = summary.get('single_run_summary', {})
        if single_summary:
            failed_count = single_summary.get('failed_models', 0)
            total_count = single_summary.get('total_models', 0)
            
            if failed_count > 0:
                failure_rate = (failed_count / total_count) * 100
                if failure_rate > 50:
                    recommendations.append(f"⚠️ High model failure rate ({failure_rate:.1f}%). Check data quality and model configurations.")
                elif failure_rate > 20:
                    recommendations.append(f"⚠️ Some models failed ({failure_rate:.1f}%). Review failed model logs.")
        
        # Rolling validation recommendations
        rolling_summary = summary.get('rolling_validation_summary', {})
        if rolling_summary:
            performance_trends = rolling_summary.get('performance_trends', {})
            
            for model, trend_info in performance_trends.items():
                trend = trend_info.get('trend', 'UNKNOWN')
                degradation = trend_info.get('degradation_percent', 0)
                
                if trend == 'SEVERE':
                    recommendations.append(f"🚨 {model} shows severe performance degradation ({degradation:.1f}%). Consider retraining or different features.")
                elif trend == 'SIGNIFICANT':
                    recommendations.append(f"⚠️ {model} shows significant performance degradation ({degradation:.1f}%). Monitor closely.")
                elif trend == 'IMPROVING':
                    recommendations.append(f"✅ {model} shows improving performance. Good model stability.")
        
        # Overall recommendations
        if summary.get('overall_best_model'):
            recommendations.append(f"🏆 Recommended model: {summary['overall_best_model']} based on overall performance.")
        
        if not recommendations:
            recommendations.append("📊 All models performed within expected ranges. Continue monitoring.")
        
        return recommendations
    
    def get_experiment_results(self) -> Dict:
        """Get current experiment results"""
        return {
            'experiment_id': self.experiment_id,
            'single_run_results': self.single_run_results,
            'rolling_results': self.rolling_results.to_dict('records') if self.rolling_results is not None else None,
            'config': self.config.to_dict()
        }
    
    def compare_with_previous_experiments(self, limit: int = 5) -> Dict:
        """Compare current results with previous experiments"""
        if not self.experiment_id:
            return {'error': 'No current experiment to compare'}
        
        # Get previous experiment results
        previous_results = self.logger_manager.get_experiment_results(limit=limit)
        
        # Get current experiment trends
        current_trends = self.logger_manager.get_performance_trends()
        
        comparison = {
            'current_experiment_id': self.experiment_id,
            'previous_experiments_count': len(previous_results),
            'performance_comparison': current_trends,
            'recommendations': []
        }
        
        # Generate comparison insights
        if current_trends:
            for model_key, trend_data in current_trends.items():
                for metric_name, metric_data in trend_data.get('metrics', {}).items():
                    if metric_name.endswith('_trend'):
                        continue
                    
                    trend_info = metric_data.get(f'{metric_name}_trend')
                    if trend_info and trend_info['count'] > 1:
                        change_pct = trend_info['change_pct']
                        if abs(change_pct) > 20:
                            direction = 'improved' if change_pct < 0 else 'degraded'
                            comparison['recommendations'].append(
                                f"{model_key} {metric_name} has {direction} by {abs(change_pct):.1f}% compared to previous runs"
                            )
        
        return comparison