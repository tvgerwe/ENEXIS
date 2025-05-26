# ============================================================================
# FILE: src/models/sarimax.py (COMPLETE)
# ============================================================================

import pandas as pd
import numpy as np
from typing import Optional, Dict
import warnings
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .factory import BaseModel
from config.experiment_config import SarimaxConfig
from core.data_manager import DataSplit

class SarimaxModel(BaseModel):
    """SARIMAX forecasting model"""
    
    def __init__(self, config: SarimaxConfig):
        super().__init__(config)
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMAX model")
        
        self.order = config.order
        self.seasonal_order = config.seasonal_order
        self.max_iterations = config.max_iterations
        self.use_exogenous = config.use_exogenous
        
        self.model = None
        self.fitted_model = None
        self.scaler = None
        self.fitted_parameters = {}
        
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