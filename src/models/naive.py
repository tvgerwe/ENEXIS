import pandas as pd
import numpy as np
from typing import Optional, Dict

from .factory import BaseModel
from config.experiment_config import NaiveConfig
from core.data_manager import DataSplit

class NaiveModel(BaseModel):
    """Naive forecasting model using seasonal lag"""
    
    def __init__(self, config: NaiveConfig):
        super().__init__(config)
        self.lag = config.lag
        self.fitted_parameters = {}
        
    def fit(self, data_split: DataSplit) -> 'NaiveModel':
        """Fit naive model (just store training data statistics)"""
        self.y_train = data_split.y_train
        
        # Store some basic statistics as "parameters"
        self.fitted_parameters = {
            'lag': self.lag,
            'train_mean': float(self.y_train.mean()),
            'train_std': float(self.y_train.std()),
            'train_min': float(self.y_train.min()),
            'train_max': float(self.y_train.max()),
            'train_samples': len(self.y_train)
        }
        
        self.is_fitted = True
        self.logger.info(f"✅ Naive model fitted with lag={self.lag}")
        return self
    
    def predict(self, data_split: DataSplit) -> pd.Series:
        """Make naive predictions using seasonal lag"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get the lag period from the end of training data
        lag_start = data_split.train_end - pd.Timedelta(hours=self.lag - 1)
        lag_end = data_split.train_end
        
        # Get historical values for the lag period
        lag_values = self.y_train.loc[lag_start:lag_end]
        
        if len(lag_values) != len(data_split.y_test):
            # If exact match not possible, use last available values
            if len(lag_values) > len(data_split.y_test):
                lag_values = lag_values.iloc[-len(data_split.y_test):]
            else:
                # Repeat last value if we don't have enough historical data
                last_value = lag_values.iloc[-1] if len(lag_values) > 0 else self.y_train.iloc[-1]
                lag_values = pd.Series(
                    [last_value] * len(data_split.y_test),
                    index=data_split.y_test.index
                )
        
        # Create predictions with the correct index
        predictions = pd.Series(
            lag_values.values,
            index=data_split.y_test.index,
            name='naive_predictions'
        )
        
        self.logger.info(f"✅ Generated {len(predictions)} naive predictions")
        return predictions
    
    def get_diagnostics(self) -> Optional[Dict]:
        """Get model diagnostics"""
        if not self.is_fitted:
            return None
        
        return {
            'model_type': 'naive',
            'lag_period': self.lag,
            'train_period_hours': len(self.y_train),
            'seasonal_pattern': 'assumed' if self.lag == 168 else 'custom'
        }
    
    def get_summary(self) -> Optional[str]:
        """Get model summary"""
        if not self.is_fitted:
            return None
        
        return f"""Naive Seasonal Model Summary:
        - Lag period: {self.lag} hours
        - Training samples: {len(self.y_train)}
        - Training period: {self.y_train.index.min()} to {self.y_train.index.max()}
        - Training mean: {self.fitted_parameters['train_mean']:.4f}
        - Training std: {self.fitted_parameters['train_std']:.4f}
        """
