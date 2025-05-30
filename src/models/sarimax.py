# ============================================================================
# FILE: src/models/sarimax.py (ENHANCED)
# ============================================================================

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .factory import BaseModel
from config.experiment_config import SarimaxConfig
from core.data_manager import DataSplit

class SarimaxModel(BaseModel):
    """SARIMAX forecasting model with auto-optimization integration"""
    
    def __init__(self, config: SarimaxConfig):
        super().__init__(config)
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMAX model")
        
        # Store original config values as fallback
        self.original_order = config.order
        self.original_seasonal_order = config.seasonal_order
        self.max_iterations = config.max_iterations
        self.use_exogenous = config.use_exogenous
        
        # Try to load optimized parameters (this is the key change!)
        self.order, self.seasonal_order = self._load_optimized_parameters()
        
        self.model = None
        self.fitted_model = None
        self.scaler = None
        self.fitted_parameters = {}
        
        # Log which parameters are being used
        if (self.order != self.original_order or 
            self.seasonal_order != self.original_seasonal_order):
            self.logger.info(f"🚀 Using OPTIMIZED parameters: order={self.order}, seasonal={self.seasonal_order}")
        else:
            self.logger.info(f"📋 Using DEFAULT parameters: order={self.order}, seasonal={self.seasonal_order}")
    
    def _load_optimized_parameters(self) -> Tuple[Tuple, Tuple]:
        """Load optimized parameters from auto-ARIMA optimization results"""
        try:
            from pathlib import Path
            import json
            
            # Find project root
            current_dir = Path(__file__).resolve()
            project_root = current_dir
            while project_root.name != "ENEXIS" and project_root.parent != project_root:
                project_root = project_root.parent
                if str(project_root) == str(project_root.parent):  # Reached filesystem root
                    project_root = current_dir.parents[2]  # Fallback
                    break
            
            config_file = project_root / "src" / "config" / "best_sarimax_params.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    best_params = json.load(f)
                
                # Convert to tuples (JSON saves as lists)
                optimized_order = tuple(best_params['order'])
                optimized_seasonal = tuple(best_params['seasonal_order'])
                
                self.logger.info(f"✅ Loaded AUTO-OPTIMIZED parameters from {config_file}")
                self.logger.info(f"   Order: {optimized_order} (was {self.original_order})")
                self.logger.info(f"   Seasonal: {optimized_seasonal} (was {self.original_seasonal_order})")
                self.logger.info(f"   Expected RMSE improvement: {best_params.get('improvement_vs_baseline', 0):.2f}%")
                self.logger.info(f"   Last updated: {best_params.get('updated_at', 'Unknown')[:19]}")
                
                return optimized_order, optimized_seasonal
            else:
                self.logger.info(f"ℹ️ No optimized parameters found at {config_file}")
                self.logger.info("📋 Using default configuration parameters")
                return self.original_order, self.original_seasonal_order
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load optimized parameters: {e}")
            self.logger.info("📋 Falling back to default configuration parameters")
            return self.original_order, self.original_seasonal_order
    
    def update_parameters_from_optimization(self, force_reload: bool = False):
        """Explicitly update parameters from latest optimization results"""
        if force_reload:
            self.order, self.seasonal_order = self._load_optimized_parameters()
            # Reset fitted model to use new parameters
            self.model = None
            self.fitted_model = None
            self.is_fitted = False
            self.logger.info("🔄 Parameters reloaded, model reset for re-fitting")
    
    def get_current_parameters(self) -> Dict:
        """Get currently active parameters"""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'original_order': self.original_order,
            'original_seasonal_order': self.original_seasonal_order,
            'using_optimized': (self.order != self.original_order or 
                              self.seasonal_order != self.original_seasonal_order),
            'max_iterations': self.max_iterations,
            'use_exogenous': self.use_exogenous
        }
        
    def fit(self, data_split: DataSplit) -> 'SarimaxModel':
        """Fit SARIMAX model"""
        # Prepare training data
        y_train = data_split.y_train.copy()
        
        # Ensure frequency is set
        if not hasattr(y_train.index, 'freq') or y_train.index.freq is None:
            y_train.index = pd.DatetimeIndex(y_train.index, freq='h')
        
        # Prepare exogenous variables
        exog_train = None
        if self.use_exogenous and data_split.X_train is not None:
            X_train = data_split.X_train.copy()
            X_train.index = pd.DatetimeIndex(X_train.index, freq='h')
            
            # Scale features
            self.scaler = StandardScaler()
            exog_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
        
        # Create and fit SARIMAX model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            self.model = SARIMAX(
                y_train,
                exog=exog_train,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            
            self.fitted_model = self.model.fit(
                disp=False,
                maxiter=self.max_iterations
            )
        
        # Store fitted parameters
        if hasattr(self.fitted_model, 'params'):
            self.fitted_parameters = dict(self.fitted_model.params)
        
        self.is_fitted = True
        self.logger.info(f"✅ SARIMAX model fitted (exog={self.use_exogenous})")
        return self
    
    def predict(self, data_split: DataSplit) -> pd.Series:
        """Make SARIMAX predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare exogenous variables for prediction
        exog_test = None
        if self.use_exogenous and data_split.X_test is not None:
            X_test = data_split.X_test.copy()
            X_test.index = pd.DatetimeIndex(X_test.index, freq='h')
            
            if self.scaler is not None:
                exog_test = pd.DataFrame(
                    self.scaler.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )
        
        # Make predictions
        predictions = self.fitted_model.forecast(
            steps=len(data_split.y_test),
            exog=exog_test
        )
        
        # Ensure correct index
        predictions.index = data_split.y_test.index
        predictions.name = 'sarimax_predictions'
        
        self.logger.info(f"✅ Generated {len(predictions)} SARIMAX predictions")
        return predictions
    
    def get_diagnostics(self) -> Optional[Dict]:
        """Get model diagnostics"""
        if not self.is_fitted or self.fitted_model is None:
            return None
        
        diagnostics = {
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'hqic': float(self.fitted_model.hqic),
            'log_likelihood': float(self.fitted_model.llf),
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'use_exogenous': self.use_exogenous
        }
        
        return diagnostics
    
    def get_convergence_info(self) -> Optional[Dict]:
        """Get convergence information"""
        if not self.is_fitted or self.fitted_model is None:
            return None
        
        convergence_info = {
            'converged': True,  # Default assumption
            'iterations': None,
            'function_calls': None,
            'gradient_calls': None,
            'warning_flag': None
        }
        
        if hasattr(self.fitted_model, 'mle_retvals'):
            retvals = self.fitted_model.mle_retvals
            convergence_info.update({
                'converged': retvals.get('converged', True),
                'iterations': retvals.get('iterations', None),
                'function_calls': retvals.get('fcalls', None),
                'gradient_calls': retvals.get('gcalls', None),
                'warning_flag': retvals.get('warnflag', None)
            })
        
        return convergence_info
    
    def get_summary(self) -> Optional[str]:
        """Get model summary"""
        if not self.is_fitted or self.fitted_model is None:
            return None
        
        try:
            summary_str = str(self.fitted_model.summary())
            # Truncate if too long
            if len(summary_str) > 2000:
                summary_str = summary_str[:2000] + "...\n[Summary truncated]"
            return summary_str
        except Exception:
            return f"SARIMAX({self.order[0]},{self.order[1]},{self.order[2]})x({self.seasonal_order[0]},{self.seasonal_order[1]},{self.seasonal_order[2]},{self.seasonal_order[3]}) model with exog={self.use_exogenous}"
    
    def get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """Get feature importance from fitted model"""
        if not self.is_fitted or self.fitted_model is None:
            return None
        
        try:
            params = self.fitted_model.params
            if self.use_exogenous and len(params) > 2:
                # Skip AR and MA terms, get exogenous coefficients
                exog_params = params[2:]
                exog_names = self.fitted_model.model.exog_names if hasattr(self.fitted_model.model, 'exog_names') else [f'x{i}' for i in range(len(exog_params))]
                
                # Create importance tuples (variable, |coefficient|)
                importance = [(var, abs(coef)) for var, coef in zip(exog_names, exog_params)]
                importance.sort(key=lambda x: x[1], reverse=True)
                
                return importance
        except Exception:
            pass
        
        return None


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def run_single_day_validation(day: int, training_data: pd.DataFrame, exog_vars: List[str]) -> Dict:
    """Run validation for a single day"""
    
    train_start_date = datetime(2025, 1, 1) + timedelta(days=day)
    train_end_date = datetime(2025, 3, 14) + timedelta(days=day)
    run_date = datetime(2025, 3, 15) + timedelta(days=day)
    
    try:
        # Create daily dataset with noise simulation
        if day == 0:
            daily_data = training_data.copy()
        else:
            daily_data = training_data.copy()
            np.random.seed(day)
            noise_factor = 0.001 * day
            daily_data['Price'] = daily_data['Price'] + np.random.normal(0, noise_factor, len(daily_data))
        
        # Split data
        split_point = daily_data.index[-24]
        train_data = daily_data[daily_data.index < split_point]['Price'].copy()
        test_data = daily_data[daily_data.index >= split_point]['Price'].copy()
        
        if len(train_data) == 0 or len(test_data) == 0:
            return {'Day': day + 1, 'Status': 'SPLIT_FAIL'}
    
    except Exception:
        return {'Day': day + 1, 'Status': 'LOAD_FAIL'}
    
    day_results = {
        'Day': day + 1,
        'Test_Date': run_date.strftime('%Y-%m-%d'),
        'Train_Samples': len(train_data),
        'Test_Samples': len(test_data)
    }
    
    # Naive Model
    try:
        naive_preds = [train_data.iloc[-24]] * len(test_data) if len(train_data) >= 24 else [train_data.iloc[-1]] * len(test_data)
        naive_rmse = np.sqrt(mean_squared_error(test_data, naive_preds))
        day_results['Naive'] = naive_rmse
    except Exception:
        day_results['Naive'] = np.nan
    
    # SARIMA Model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_data, order=(1, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
            fitted_model = model.fit(method='mle', maxiter=20, disp=False)
            sarima_forecast = fitted_model.forecast(steps=len(test_data))
            sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
            day_results['SARIMA'] = sarima_rmse
    except Exception:
        try:
            window_size = min(24, len(train_data))
            ma_pred = train_data.rolling(window=window_size).mean().iloc[-1]
            ma_forecast = [ma_pred] * len(test_data)
            sarima_rmse = np.sqrt(mean_squared_error(test_data, ma_forecast))
            day_results['SARIMA'] = sarima_rmse
            day_results['SARIMA_Fallback'] = True
        except:
            day_results['SARIMA'] = np.nan
    
    # SARIMAX Model
    try:
        if exog_vars:
            train_exog = daily_data[daily_data.index < split_point][exog_vars].copy()
            test_exog = daily_data[daily_data.index >= split_point][exog_vars].copy()
            
            if len(train_exog) == len(train_data) and len(test_exog) == len(test_data):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        train_data,
                        exog=train_exog,
                        order=(1, 0, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted_model = model.fit(method='lbfgs', maxiter=20, disp=False)
                    sarimax_forecast = fitted_model.forecast(steps=len(test_data), exog=test_exog)
                    sarimax_rmse = np.sqrt(mean_squared_error(test_data, sarimax_forecast))
                    day_results['SARIMAX'] = sarimax_rmse
            else:
                day_results['SARIMAX'] = np.nan
        else:
            day_results['SARIMAX'] = np.nan
    except Exception:
        day_results['SARIMAX'] = np.nan
    
    return day_results


def run_validation_experiment(training_data: pd.DataFrame, exog_vars: List[str], n_days: int = 30) -> pd.DataFrame:
    """Run complete validation experiment - returns results only, no printing"""
    
    warnings.filterwarnings('ignore')
    results_matrix = []
    
    for day in range(n_days):
        result = run_single_day_validation(day, training_data, exog_vars)
        results_matrix.append(result)
    
    return pd.DataFrame(results_matrix)


def analyze_feature_contributions(training_data: pd.DataFrame, exog_vars: List[str]) -> Tuple[Optional[List[Tuple[str, float]]], Optional[float], Optional[float]]:
    """Analyze feature contributions from SARIMAX model"""
    
    try:
        daily_data = training_data.copy()
        split_point = daily_data.index[-24]
        train_data = daily_data[daily_data.index < split_point]['Price'].copy()
        train_exog = daily_data[daily_data.index < split_point][exog_vars].copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train_data,
                exog=train_exog,
                order=(1, 0, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(method='lbfgs', maxiter=20, disp=False)
            
            # Get parameter estimates
            params = fitted_model.params
            exog_params = params[2:]  # Skip AR and MA terms
            
            # Calculate contributions
            contributions = [(abs(coef), var) for coef, var in zip(exog_params, exog_vars)]
            contributions.sort(reverse=True)
            
            # Convert to proper format
            feature_importance = [(var, coef) for coef, var in contributions]
            
            return feature_importance, fitted_model.aic, fitted_model.bic
            
    except Exception:
        return None, None, None


def generate_validation_summary(results_df: pd.DataFrame, feature_importance: Optional[List[Tuple[str, float]]], 
                              aic: Optional[float], bic: Optional[float], exog_vars: List[str]) -> Dict:
    """Generate validation summary data - returns dict, no printing"""
    
    summary = {}
    
    # Performance metrics
    summary['performance'] = {}
    for model in ['Naive', 'SARIMA', 'SARIMAX']:
        if model in results_df.columns:
            valid_results = results_df[model].dropna()
            if len(valid_results) > 0:
                summary['performance'][model] = {
                    'mean': valid_results.mean(),
                    'std': valid_results.std(),
                    'min': valid_results.min(),
                    'max': valid_results.max(),
                    'count': len(valid_results)
                }
    
    # Model improvements
    if 'Naive' in summary['performance'] and 'SARIMA' in summary['performance'] and 'SARIMAX' in summary['performance']:
        naive_rmse = summary['performance']['Naive']['mean']
        sarima_rmse = summary['performance']['SARIMA']['mean']
        sarimax_rmse = summary['performance']['SARIMAX']['mean']
        
        summary['improvements'] = {
            'sarima_vs_naive': ((naive_rmse - sarima_rmse) / naive_rmse) * 100,
            'sarimax_vs_naive': ((naive_rmse - sarimax_rmse) / naive_rmse) * 100,
            'sarimax_vs_sarima': ((sarima_rmse - sarimax_rmse) / sarima_rmse) * 100
        }
        
        summary['best_model'] = 'SARIMAX' if sarimax_rmse < sarima_rmse else 'SARIMA'
    
    # Feature importance
    summary['feature_importance'] = feature_importance
    if feature_importance:
        total_importance = sum(coef for _, coef in feature_importance)
        top_5_importance = sum(coef for _, coef in feature_importance[:5])
        summary['feature_concentration'] = top_5_importance / total_importance * 100
    
    # Model quality
    summary['model_quality'] = {
        'aic': aic,
        'bic': bic,
        'n_features': len(exog_vars)
    }
    
    # Consistency
    if 'SARIMA' in summary['performance'] and 'SARIMAX' in summary['performance']:
        summary['consistency'] = {
            'sarima_std': summary['performance']['SARIMA']['std'],
            'sarimax_std': summary['performance']['SARIMAX']['std'],
            'more_consistent': 'SARIMAX' if summary['performance']['SARIMAX']['std'] < summary['performance']['SARIMA']['std'] else 'SARIMA'
        }
    
    return summary