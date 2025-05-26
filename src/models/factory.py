import logging
from typing import Dict, List, Optional, Type
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time

from config.experiment_config import ExperimentConfig, ModelConfig
from core.data_manager import DataSplit

@dataclass
class ModelResult:
    """Standardized model result container"""
    predictions: pd.Series
    model_name: str
    model_variant: str
    execution_time: float
    parameters: Dict
    hyperparameters: Dict
    diagnostics: Optional[Dict] = None
    convergence_info: Optional[Dict] = None
    model_summary: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error_message is None and self.predictions is not None

class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data_split: DataSplit) -> 'BaseModel':
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict(self, data_split: DataSplit) -> pd.Series:
        """Make predictions"""
        pass
    
    def fit_predict(self, data_split: DataSplit) -> ModelResult:
        """Fit model and make predictions with timing and error handling"""
        start_time = time.time()
        
        try:
            # Fit model
            self.fit(data_split)
            
            # Make predictions
            predictions = self.predict(data_split)
            
            execution_time = time.time() - start_time
            
            # Get model information
            diagnostics = self.get_diagnostics() if hasattr(self, 'get_diagnostics') else None
            convergence_info = self.get_convergence_info() if hasattr(self, 'get_convergence_info') else None
            model_summary = self.get_summary() if hasattr(self, 'get_summary') else None
            
            return ModelResult(
                predictions=predictions,
                model_name=self.config.name,
                model_variant=self.config.name,
                execution_time=execution_time,
                parameters=getattr(self, 'fitted_parameters', {}),
                hyperparameters=self.config.hyperparameters,
                diagnostics=diagnostics,
                convergence_info=convergence_info,
                model_summary=model_summary
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Model {self.config.name} failed: {e}")
            
            return ModelResult(
                predictions=None,
                model_name=self.config.name,
                model_variant=self.config.name,
                execution_time=execution_time,
                parameters={},
                hyperparameters=self.config.hyperparameters,
                error_message=str(e)
            )

class ModelFactory:
    """Factory for creating and managing models"""
    
    def __init__(self, model_configs: Dict[str, ModelConfig]):
        self.model_configs = model_configs
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model_registry = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model types"""
        from .naive import NaiveModel
        from .sarimax import SarimaxModel
        
        self._model_registry['naive'] = NaiveModel
        self._model_registry['sarimax_no_exog'] = SarimaxModel
        self._model_registry['sarimax_with_exog'] = SarimaxModel
    
    def register_model(self, model_name: str, model_class: Type[BaseModel]):
        """Register a new model type"""
        self._model_registry[model_name] = model_class
        self.logger.info(f"✅ Registered model type: {model_name}")
    
    def create_model(self, model_name: str) -> BaseModel:
        """Create a model instance"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        if model_name not in self._model_registry:
            raise ValueError(f"Model type '{model_name}' not registered")
        
        config = self.model_configs[model_name]
        if not config.enabled:
            raise ValueError(f"Model '{model_name}' is disabled")
        
        model_class = self._model_registry[model_name]
        return model_class(config)
    
    def create_all_models(self) -> Dict[str, BaseModel]:
        """Create all enabled models"""
        models = {}
        for model_name, config in self.model_configs.items():
            if config.enabled:
                try:
                    models[model_name] = self.create_model(model_name)
                    self.logger.info(f"✅ Created model: {model_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to create model {model_name}: {e}")
        return models
    
    def run_single_model(self, model_name: str, data_split: DataSplit) -> ModelResult:
        """Run a single model"""
        model = self.create_model(model_name)
        return model.fit_predict(data_split)
    
    def run_all_models(self, data_split: DataSplit, parallel: bool = False) -> Dict[str, ModelResult]:
        """Run all enabled models"""
        models = self.create_all_models()
        results = {}
        
        if parallel and len(models) > 1:
            results = self._run_models_parallel(models, data_split)
        else:
            results = self._run_models_sequential(models, data_split)
        
        return results
    
    def _run_models_sequential(self, models: Dict[str, BaseModel], data_split: DataSplit) -> Dict[str, ModelResult]:
        """Run models sequentially"""
        results = {}
        for model_name, model in models.items():
            self.logger.info(f"🏃 Running model: {model_name}")
            results[model_name] = model.fit_predict(data_split)
            
            if results[model_name].success:
                self.logger.info(f"✅ {model_name} completed in {results[model_name].execution_time:.2f}s")
            else:
                self.logger.error(f"❌ {model_name} failed: {results[model_name].error_message}")
        
        return results
    
    def _run_models_parallel(self, models: Dict[str, BaseModel], data_split: DataSplit) -> Dict[str, ModelResult]:
        """Run models in parallel"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        max_workers = min(len(models), mp.cpu_count())
        results = {}
        
        self.logger.info(f"🚀 Running {len(models)} models in parallel with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_model = {}
            for model_name, model in models.items():
                future = executor.submit(model.fit_predict, data_split)
                future_to_model[future] = model_name
            
            # Collect results
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results[model_name] = result
                    
                    if result.success:
                        self.logger.info(f"✅ {model_name} completed in {result.execution_time:.2f}s")
                    else:
                        self.logger.error(f"❌ {model_name} failed: {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {model_name} execution failed: {e}")
                    results[model_name] = ModelResult(
                        predictions=None,
                        model_name=model_name,
                        model_variant=model_name,
                        execution_time=0,
                        parameters={},
                        hyperparameters={},
                        error_message=str(e)
                    )
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about available models"""
        info = {
            'total_models': len(self.model_configs),
            'enabled_models': [name for name, config in self.model_configs.items() if config.enabled],
            'disabled_models': [name for name, config in self.model_configs.items() if not config.enabled],
            'registered_types': list(self._model_registry.keys()),
            'model_details': {}
        }
        
        for name, config in self.model_configs.items():
            info['model_details'][name] = {
                'enabled': config.enabled,
                'hyperparameters': config.hyperparameters,
                'type': type(config).__name__
            }
        
        return info