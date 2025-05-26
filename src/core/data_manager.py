# ============================================================================
# FILE: src/core/data_manager.py
# ============================================================================

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager

from config.experiment_config import ExperimentConfig

@dataclass
class DataSplit:
    """Container for data splits"""
    y_train: pd.Series
    X_train: Optional[pd.DataFrame]
    y_test: pd.Series
    X_test: Optional[pd.DataFrame]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    
    def __len__(self):
        return len(self.y_train)
    
    def get_info(self) -> Dict:
        """Get information about this data split"""
        return {
            'train_samples': len(self.y_train),
            'test_samples': len(self.y_test),
            'train_period': f"{self.train_start} to {self.train_end}",
            'forecast_period': f"{self.forecast_start} to {self.forecast_end}",
            'features': list(self.X_train.columns) if self.X_train is not None else [],
            'target_stats': {
                'train_mean': self.y_train.mean(),
                'train_std': self.y_train.std(),
                'test_mean': self.y_test.mean(),
                'test_std': self.y_test.std()
            }
        }

class DataManager:
    """Unified data management for experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._data_cache = {}
        
    @contextmanager
    def _get_connection(self, db_path: Optional[Path] = None):
        """Context manager for database connections"""
        path = db_path or self.config.database_path
        conn = sqlite3.connect(path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _load_table(self, table_name: str, query: Optional[str] = None) -> pd.DataFrame:
        """Load table with caching"""
        cache_key = f"{table_name}_{hash(query) if query else 'full'}"
        
        if cache_key not in self._data_cache:
            with self._get_connection() as conn:
                if query:
                    df = pd.read_sql_query(query, conn)
                else:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
            self._data_cache[cache_key] = df
            self.logger.info(f"✅ Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
        
        return self._data_cache[cache_key].copy()
    
    def get_master_data(self, use_training_set: bool = True) -> pd.DataFrame:
        """Get the master dataset (either training_set or master_warp)"""
        table_name = "training_set" if use_training_set else "master_warp"
        
        try:
            df = self._load_table(table_name)
        except Exception as e:
            if use_training_set:
                self.logger.warning(f"Training set not found, building it: {e}")
                self.build_training_set()
                df = self._load_table(table_name)
            else:
                raise e
        
        # Ensure proper datetime and timezone handling
        df["target_datetime"] = pd.to_datetime(df["target_datetime"], utc=True)
        df = df.sort_values("target_datetime").set_index("target_datetime")
        df = df[~df.index.duplicated(keep='first')]
        
        # Validate required columns
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        
        missing_features = [col for col in self.config.feature_columns if col not in df.columns]
        if missing_features:
            self.logger.warning(f"Missing feature columns: {missing_features}")
        
        return df
    
    def create_splits(self, 
                     train_start: Optional[pd.Timestamp] = None,
                     train_end: Optional[pd.Timestamp] = None,
                     forecast_start: Optional[pd.Timestamp] = None,
                     forecast_horizon: Optional[int] = None,
                     use_training_set: bool = True) -> DataSplit:
        """Create train/test splits"""
        
        # Use config defaults if not provided
        train_start = train_start or self.config.train_start
        train_end = train_end or self.config.train_end
        forecast_start = forecast_start or self.config.forecast_start
        forecast_horizon = forecast_horizon or self.config.horizon
        
        forecast_end = forecast_start + pd.Timedelta(hours=forecast_horizon - 1)
        
        # Load data
        df = self.get_master_data(use_training_set=use_training_set)
        
        # Create target series
        y = df[self.config.target_column].dropna()
        
        # Create feature matrix
        available_features = [col for col in self.config.feature_columns if col in df.columns]
        if available_features:
            X = df[available_features].loc[y.index]
        else:
            X = None
            self.logger.warning("No feature columns available")
        
        # Create splits
        y_train = y.loc[train_start:train_end]
        y_test = y.loc[forecast_start:forecast_end]
        
        if X is not None:
            X_train = X.loc[train_start:train_end]
            X_test = X.loc[forecast_start:forecast_end]
        else:
            X_train = X_test = None
        
        # Validate splits
        if len(y_train) == 0:
            raise ValueError(f"No training data found between {train_start} and {train_end}")
        if len(y_test) == 0:
            raise ValueError(f"No test data found between {forecast_start} and {forecast_end}")
        
        split = DataSplit(
            y_train=y_train,
            X_train=X_train,
            y_test=y_test,
            X_test=X_test,
            train_start=train_start,
            train_end=train_end,
            forecast_start=forecast_start,
            forecast_end=forecast_end
        )
        
        self.logger.info(f"✅ Created data split: {len(y_train)} train, {len(y_test)} test samples")
        return split
    
    def create_rolling_splits(self, n_windows: int = 3) -> List[DataSplit]:
        """Create multiple rolling window splits"""
        splits = []
        
        for i in range(n_windows):
            delta = pd.Timedelta(days=i)
            train_start_i = self.config.train_start + delta
            train_end_i = self.config.train_end + delta
            forecast_start_i = train_end_i + pd.Timedelta(hours=1)
            
            try:
                split = self.create_splits(
                    train_start=train_start_i,
                    train_end=train_end_i,
                    forecast_start=forecast_start_i
                )
                splits.append(split)
                self.logger.info(f"✅ Created rolling window {i+1}/{n_windows}")
            except ValueError as e:
                self.logger.warning(f"Skipping rolling window {i+1}: {e}")
                continue
        
        return splits
    
    def build_training_set(self) -> bool:
        """Build training set by combining actuals and predictions"""
        try:
            self.logger.info("🚀 Building training set...")
            
            # Calculate periods
            train_start = self.config.train_start
            train_end = self.config.train_end
            run_date = train_end + pd.Timedelta(hours=1)
            forecast_start = run_date
            forecast_end = forecast_start + pd.Timedelta(hours=self.config.horizon - 1)
            
            with self._get_connection() as conn:
                # Load actuals
                df_actuals = self._load_table("master_warp")
                df_actuals["target_datetime"] = pd.to_datetime(df_actuals["target_datetime"], utc=True)
                df_actuals = df_actuals[
                    (df_actuals["target_datetime"] >= train_start) &
                    (df_actuals["target_datetime"] <= train_end)
                ]
                
                # Drop problematic columns
                columns_to_drop = ['wind_direction_10m', 'direct_radiation']
                existing_columns = [col for col in columns_to_drop if col in df_actuals.columns]
                if existing_columns:
                    df_actuals = df_actuals.drop(columns=existing_columns)
                    self.logger.info(f"🗑️ Dropped columns: {existing_columns}")
                
                self.logger.info(f"✅ Loaded actuals: {len(df_actuals)} rows")
                
                # Load predictions
                df_preds = self._load_table("master_predictions")
                df_preds["target_datetime"] = pd.to_datetime(df_preds["target_datetime"], utc=True)
                df_preds["run_date"] = pd.to_datetime(df_preds["run_date"], utc=True)
                
                # Drop same problematic columns
                existing_columns = [col for col in columns_to_drop if col in df_preds.columns]
                if existing_columns:
                    df_preds = df_preds.drop(columns=existing_columns)
                
                # Filter predictions by run_date and target_datetime
                run_date_only = run_date.date()
                df_preds_filtered = df_preds[
                    (df_preds["run_date"].dt.date == run_date_only) &
                    (df_preds["target_datetime"] >= forecast_start) &
                    (df_preds["target_datetime"] <= forecast_end)
                ]
                
                # If no exact match, find closest run_date
                if df_preds_filtered.empty:
                    self.logger.warning("⚠️ No exact run_date match, finding closest...")
                    closest_before = df_preds[df_preds["run_date"] <= run_date]
                    if not closest_before.empty:
                        max_run_date = closest_before["run_date"].max()
                        df_preds_filtered = df_preds[
                            (df_preds["run_date"] == max_run_date) &
                            (df_preds["target_datetime"] >= forecast_start) &
                            (df_preds["target_datetime"] <= forecast_end)
                        ]
                        self.logger.info(f"📅 Using run_date: {max_run_date}")
                
                self.logger.info(f"✅ Loaded predictions: {len(df_preds_filtered)} rows")
                
                # Remove run_date column from predictions to avoid conflicts
                if "run_date" in df_preds_filtered.columns:
                    df_preds_filtered = df_preds_filtered.drop(columns=["run_date"])
                
                # Combine datasets
                df_combined = pd.concat([df_actuals, df_preds_filtered], ignore_index=True)
                df_combined = df_combined.sort_values("target_datetime")
                df_combined = df_combined.drop_duplicates("target_datetime", keep='first')
                
                self.logger.info(f"📦 Combined dataset: {len(df_combined)} rows, {len(df_combined.columns)} columns")
                
                # Save to database
                df_combined.to_sql("training_set", conn, if_exists="replace", index=False)
                self.logger.info("✅ Training set saved successfully")
                
                # Clear cache to force reload
                self._data_cache = {}
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to build training set: {e}", exc_info=True)
            return False
    
    def validate_data_quality(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Validate data quality and return report"""
        if df is None:
            df = self.get_master_data()
        
        report = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min().isoformat() if len(df) > 0 else None,
                'end': df.index.max().isoformat() if len(df) > 0 else None
            },
            'missing_data': {},
            'duplicate_timestamps': df.index.duplicated().sum(),
            'target_column_stats': {},
            'feature_column_stats': {}
        }
        
        # Check missing data
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # Target column statistics
        if self.config.target_column in df.columns:
            target_series = df[self.config.target_column]
            report['target_column_stats'] = {
                'mean': target_series.mean(),
                'std': target_series.std(),
                'min': target_series.min(),
                'max': target_series.max(),
                'missing_count': target_series.isna().sum()
            }
        
        # Feature column statistics
        available_features = [col for col in self.config.feature_columns if col in df.columns]
        for col in available_features:
            series = df[col]
            report['feature_column_stats'][col] = {
                'mean': series.mean(),
                'std': series.std(),
                'missing_count': series.isna().sum()
            }
        
        # Data quality score
        missing_score = 100 - (sum(info['percentage'] for info in report['missing_data'].values()) / len(df.columns))
        duplicate_score = 100 - ((report['duplicate_timestamps'] / len(df)) * 100) if len(df) > 0 else 100
        report['quality_score'] = (missing_score + duplicate_score) / 2
        
        return report
    
    def get_data_info(self) -> Dict:
        """Get comprehensive data information"""
        try:
            df = self.get_master_data()
            quality_report = self.validate_data_quality(df)
            
            return {
                'data_source': 'training_set',
                'shape': df.shape,
                'columns': list(df.columns),
                'target_column': self.config.target_column,
                'feature_columns': [col for col in self.config.feature_columns if col in df.columns],
                'missing_features': [col for col in self.config.feature_columns if col not in df.columns],
                'quality_report': quality_report
            }
        except Exception as e:
            self.logger.error(f"Error getting data info: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear data cache"""
        self._data_cache = {}
        self.logger.info("🗑️ Data cache cleared")