import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class MetricsCalculator:
    """Standardized metrics calculation for model evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_rmse(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Root Mean Square Error"""
        try:
            # Align series by index
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) == 0:
                return np.nan
            
            y_true_aligned = y_true.loc[common_idx].dropna()
            y_pred_aligned = y_pred.loc[common_idx].dropna()
            
            # Get common indices after dropping NaN
            final_common_idx = y_true_aligned.index.intersection(y_pred_aligned.index)
            if len(final_common_idx) == 0:
                return np.nan
            
            y_true_final = y_true_aligned.loc[final_common_idx]
            y_pred_final = y_pred_aligned.loc[final_common_idx]
            
            return np.sqrt(np.mean((y_true_final - y_pred_final) ** 2))
            
        except Exception as e:
            self.logger.warning(f"Error calculating RMSE: {e}")
            return np.nan
    
    def calculate_mae(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Mean Absolute Error"""
        try:
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) == 0:
                return np.nan
            
            y_true_aligned = y_true.loc[common_idx].dropna()
            y_pred_aligned = y_pred.loc[common_idx].dropna()
            
            final_common_idx = y_true_aligned.index.intersection(y_pred_aligned.index)
            if len(final_common_idx) == 0:
                return np.nan
            
            y_true_final = y_true_aligned.loc[final_common_idx]
            y_pred_final = y_pred_aligned.loc[final_common_idx]
            
            return np.mean(np.abs(y_true_final - y_pred_final))
            
        except Exception as e:
            self.logger.warning(f"Error calculating MAE: {e}")
            return np.nan
    
    def calculate_mape(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Mean Absolute Percentage Error"""
        try:
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) == 0:
                return np.nan
            
            y_true_aligned = y_true.loc[common_idx].dropna()
            y_pred_aligned = y_pred.loc[common_idx].dropna()
            
            final_common_idx = y_true_aligned.index.intersection(y_pred_aligned.index)
            if len(final_common_idx) == 0:
                return np.nan
            
            y_true_final = y_true_aligned.loc[final_common_idx]
            y_pred_final = y_pred_aligned.loc[final_common_idx]
            
            # Avoid division by zero
            mask = y_true_final != 0
            if mask.sum() == 0:
                return np.nan
            
            return np.mean(np.abs((y_true_final[mask] - y_pred_final[mask]) / y_true_final[mask])) * 100
            
        except Exception as e:
            self.logger.warning(f"Error calculating MAPE: {e}")
            return np.nan
    
    def calculate_all_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate all standard metrics"""
        return {
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mae': self.calculate_mae(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred)
        }
    
    def calculate_detailed_rmse(self, y_true: pd.Series, y_pred: pd.Series) -> Dict:
        """Calculate detailed RMSE breakdown (overall, per day, per hour)"""
        try:
            # Overall RMSE
            overall_rmse = self.calculate_rmse(y_true, y_pred)
            
            # Align data
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) == 0:
                return {'overall': np.nan, 'per_day': {}, 'per_hour': {}}
            
            df = pd.DataFrame({
                'actual': y_true.loc[common_idx],
                'predicted': y_pred.loc[common_idx]
            }).dropna()
            
            if len(df) == 0:
                return {'overall': overall_rmse, 'per_day': {}, 'per_hour': {}}
            
            # Per day RMSE
            df['date'] = df.index.date
            daily_rmse = df.groupby('date').apply(
                lambda x: np.sqrt(np.mean((x['actual'] - x['predicted']) ** 2))
            ).dropna()
            
            rmse_per_day = {}
            for i, (date, rmse_val) in enumerate(daily_rmse.items(), 1):
                if i <= 7:  # Limit to 7 days
                    rmse_per_day[str(i)] = round(float(rmse_val), 6)
            
            # Per hour absolute errors (not RMSE)
            hourly_errors = np.abs(df['actual'] - df['predicted'])
            rmse_per_hour = {}
            for i, error in enumerate(hourly_errors.values[:168]):  # Limit to 168 hours
                rmse_per_hour[str(i)] = round(float(error), 6)
            
            return {
                'overall': round(float(overall_rmse), 6),
                'per_day': rmse_per_day,
                'per_hour': rmse_per_hour
            }
            
        except Exception as e:
            self.logger.error(f"Error in detailed RMSE calculation: {e}")
            return {'overall': np.nan, 'per_day': {}, 'per_hour': {}}
    
    def calculate_statistical_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict:
        """Calculate additional statistical metrics"""
        try:
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) == 0:
                return {}
            
            y_true_aligned = y_true.loc[common_idx].dropna()
            y_pred_aligned = y_pred.loc[common_idx].dropna()
            
            final_common_idx = y_true_aligned.index.intersection(y_pred_aligned.index)
            if len(final_common_idx) == 0:
                return {}
            
            y_true_final = y_true_aligned.loc[final_common_idx]
            y_pred_final = y_pred_aligned.loc[final_common_idx]
            
            # Calculate various metrics
            residuals = y_true_final - y_pred_final
            
            metrics = {
                'mean_residual': float(residuals.mean()),
                'std_residual': float(residuals.std()),
                'min_residual': float(residuals.min()),
                'max_residual': float(residuals.max()),
                'correlation': float(np.corrcoef(y_true_final, y_pred_final)[0, 1]),
                'r_squared': float(np.corrcoef(y_true_final, y_pred_final)[0, 1] ** 2),
                'mean_absolute_residual': float(np.abs(residuals).mean()),
                'median_absolute_residual': float(np.abs(residuals).median())
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical metrics: {e}")
            return {}
    
    def calculate_data_statistics(self, series: pd.Series, name: str = "data") -> Dict:
        """Calculate descriptive statistics for a data series"""
        try:
            clean_series = series.dropna()
            if len(clean_series) == 0:
                return {}
            
            return {
                f'{name}_count': len(clean_series),
                f'{name}_mean': float(clean_series.mean()),
                f'{name}_std': float(clean_series.std()),
                f'{name}_min': float(clean_series.min()),
                f'{name}_max': float(clean_series.max()),
                f'{name}_median': float(clean_series.median()),
                f'{name}_q25': float(clean_series.quantile(0.25)),
                f'{name}_q75': float(clean_series.quantile(0.75)),
                f'{name}_missing_count': len(series) - len(clean_series),
                f'{name}_missing_pct': ((len(series) - len(clean_series)) / len(series)) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics for {name}: {e}")
            return {}
    
    def compare_predictions(self, y_true: pd.Series, predictions_dict: Dict[str, pd.Series]) -> Dict:
        """Compare multiple model predictions"""
        comparison = {
            'models': list(predictions_dict.keys()),
            'metrics_comparison': {},
            'ranking': {}
        }
        
        # Calculate metrics for each model
        for model_name, y_pred in predictions_dict.items():
            if y_pred is not None:
                metrics = self.calculate_all_metrics(y_true, y_pred)
                statistical_metrics = self.calculate_statistical_metrics(y_true, y_pred)
                
                comparison['metrics_comparison'][model_name] = {
                    **metrics,
                    **statistical_metrics
                }
        
        # Rank models by RMSE (lower is better)
        valid_rmse = {
            model: metrics['rmse'] 
            for model, metrics in comparison['metrics_comparison'].items() 
            if not np.isnan(metrics.get('rmse', np.nan))
        }
        
        if valid_rmse:
            sorted_models = sorted(valid_rmse.items(), key=lambda x: x[1])
            comparison['ranking'] = {
                'by_rmse': [{'model': model, 'rmse': rmse} for model, rmse in sorted_models],
                'best_model': sorted_models[0][0],
                'worst_model': sorted_models[-1][0]
            }
        
        return comparison
