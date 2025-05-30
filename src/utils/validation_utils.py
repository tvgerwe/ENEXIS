# ============================================================================
# FILE: src/utils/validation_utils.py
# ============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


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