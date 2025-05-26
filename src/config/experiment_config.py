# src/config/experiment_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import yaml
import json

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    enabled: bool = True
    parameters: Dict = field(default_factory=dict)
    hyperparameters: Dict = field(default_factory=dict)

@dataclass 
class SarimaxConfig(ModelConfig):
    """SARIMAX-specific configuration"""
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    max_iterations: int = 100
    use_exogenous: bool = True
    
    def __post_init__(self):
        self.hyperparameters.update({
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'max_iterations': self.max_iterations,
            'use_exogenous': self.use_exogenous
        })

@dataclass
class NaiveConfig(ModelConfig):
    """Naive model configuration"""
    lag: int = 168
    
    def __post_init__(self):
        self.hyperparameters.update({
            'lag': self.lag
        })

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    # Data paths
    database_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "WARP.db")
    logs_database_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "logs.db")
    
    # Time periods
    train_start: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    train_end: pd.Timestamp = pd.Timestamp("2025-03-14 23:00:00", tz="UTC")
    forecast_start: pd.Timestamp = pd.Timestamp("2025-03-15 00:00:00", tz="UTC")
    horizon: int = 168
    
    # Feature configuration
    target_column: str = "Price"
    feature_columns: List[str] = field(default_factory=lambda: [
        "Load", "shortwave_radiation", "temperature_2m", "direct_normal_irradiance", 
        "diffuse_radiation", "Flow_NO", "yearday_cos", "Flow_GB", "month", "is_dst", 
        "yearday_sin", "is_non_working_day", "hour_cos", "is_weekend", "cloud_cover", 
        "weekday_sin", "hour_sin", "weekday_cos"
    ])
    
    # Model configurations
    model_configs: Dict[str, ModelConfig] = field(default_factory=lambda: {
        'naive': NaiveConfig(name='naive', lag=168),
        'sarimax_no_exog': SarimaxConfig(
            name='sarimax_no_exog', 
            use_exogenous=False
        ),
        'sarimax_with_exog': SarimaxConfig(
            name='sarimax_with_exog', 
            use_exogenous=True
        )
    })
    
    # Validation settings
    rolling_windows: int = 3
    parallel_execution: bool = False
    max_workers: int = 4
    
    # Logging settings
    log_level: str = "INFO"
    save_detailed_logs: bool = True
    save_model_summaries: bool = True
    
    @property
    def forecast_end(self) -> pd.Timestamp:
        """Calculate forecast end based on start and horizon"""
        return self.forecast_start + pd.Timedelta(hours=self.horizon - 1)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Convert timestamp strings to pandas Timestamps
        for time_field in ['train_start', 'train_end', 'forecast_start']:
            if time_field in config_dict:
                config_dict[time_field] = pd.Timestamp(config_dict[time_field], tz="UTC")
        
        # Convert paths to Path objects
        for path_field in ['database_path', 'logs_database_path']:
            if path_field in config_dict:
                config_dict[path_field] = Path(config_dict[path_field])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return {
            'database_path': str(self.database_path),
            'logs_database_path': str(self.logs_database_path),
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'forecast_start': self.forecast_start.isoformat(),
            'forecast_end': self.forecast_end.isoformat(),
            'horizon': self.horizon,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'model_configs': {k: v.hyperparameters for k, v in self.model_configs.items()},
            'rolling_windows': self.rolling_windows,
            'parallel_execution': self.parallel_execution
        }
