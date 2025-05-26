# ============================================================================
# FILE: src/visualization/results.py
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any, Tuple
from IPython.display import display, HTML

from models.factory import ModelResult
from evaluation.metrics import MetricsCalculator

class ResultsVisualizer:
    """Visualization utilities for experiment results"""
    
    def __init__(self, use_plotly: bool = True):
        self.use_plotly = use_plotly
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_calculator = MetricsCalculator()
        
        # Color palette for consistent model colors
        self.model_colors = {
            'naive': '#ff7f0e',
            'sarimax_no_exog': '#1f77b4', 
            'sarimax_with_exog': '#2ca02c',
            'arima': '#d62728',
            'prophet': '#9467bd',
            'lstm': '#8c564b',
            'xgboost': '#e377c2'
        }
        
        # Line styles for differentiation
        self.line_styles = {
            'naive': 'dash',
            'sarimax_no_exog': 'dot',
            'sarimax_with_exog': 'dashdot',
            'arima': 'solid',
            'prophet': 'longdash',
            'lstm': 'longdashdot',
            'xgboost': 'solid'
        }
    
    def create_comparison_plot(self, 
                             actual_values: pd.Series,
                             model_results: Dict[str, ModelResult],
                             training_data: Optional[pd.Series] = None,
                             title: str = "Model Comparison",
                             show_training: bool = True) -> go.Figure:
        """Create interactive comparison plot of model predictions"""
        
        if self.use_plotly:
            return self._create_plotly_comparison(
                actual_values, model_results, training_data, title, show_training
            )
        else:
            return self._create_matplotlib_comparison(
                actual_values, model_results, training_data, title, show_training
            )
    
    def _create_plotly_comparison(self,
                                actual_values: pd.Series,
                                model_results: Dict[str, ModelResult], 
                                training_data: Optional[pd.Series] = None,
                                title: str = "Model Comparison",
                                show_training: bool = True) -> go.Figure:
        """Create Plotly comparison plot"""
        
        fig = go.Figure()
        
        # Add training data if provided
        if show_training and training_data is not None:
            fig.add_trace(go.Scatter(
                x=training_data.index,
                y=training_data.values,
                mode='lines',
                name='Training Data',
                line=dict(color='lightgray', width=1),
                opacity=0.7
            ))
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=actual_values.index,
            y=actual_values.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Add model predictions
        for i, (model_name, result) in enumerate(model_results.items()):
            if result.success and result.predictions is not None:
                # Calculate RMSE for legend
                rmse = self.metrics_calculator.calculate_rmse(actual_values, result.predictions)
                
                # Get model color and style
                color = self.model_colors.get(model_name, f'hsl({i * 360 / len(model_results)}, 70%, 50%)')
                dash = self.line_styles.get(model_name, 'solid')
                
                fig.add_trace(go.Scatter(
                    x=result.predictions.index,
                    y=result.predictions.values,
                    mode='lines',
                    name=f'{model_name.replace("_", " ").title()} (RMSE: {rmse:.4f})',
                    line=dict(color=color, dash=dash, width=2)
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time (UTC)',
            yaxis_title='Price',
            template='plotly_white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def _create_matplotlib_comparison(self,
                                    actual_values: pd.Series,
                                    model_results: Dict[str, ModelResult],
                                    training_data: Optional[pd.Series] = None,
                                    title: str = "Model Comparison",
                                    show_training: bool = True) -> plt.Figure:
        """Create Matplotlib comparison plot"""
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Add training data
        if show_training and training_data is not None:
            ax.plot(training_data.index, training_data.values, 
                   color='lightgray', alpha=0.7, label='Training Data')
        
        # Add actual values
        ax.plot(actual_values.index, actual_values.values, 
               color='black', linewidth=2, label='Actual')
        
        # Add model predictions
        for model_name, result in model_results.items():
            if result.success and result.predictions is not None:
                rmse = self.metrics_calculator.calculate_rmse(actual_values, result.predictions)
                color = self.model_colors.get(model_name, None)
                
                ax.plot(result.predictions.index, result.predictions.values,
                       color=color, linewidth=1.5,
                       label=f'{model_name.replace("_", " ").title()} (RMSE: {rmse:.4f})')
        
        ax.set_title(title)
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_summary(self, 
                                 actual_values: pd.Series,
                                 model_results: Dict[str, ModelResult]) -> pd.DataFrame:
        """Create performance summary table"""
        
        summary_data = []
        
        for model_name, result in model_results.items():
            if result.success and result.predictions is not None:
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(actual_values, result.predictions)
                statistical_metrics = self.metrics_calculator.calculate_statistical_metrics(actual_values, result.predictions)
                
                summary_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'RMSE': f"{metrics.get('rmse', np.nan):.6f}",
                    'MAE': f"{metrics.get('mae', np.nan):.6f}",
                    'MAPE': f"{metrics.get('mape', np.nan):.2f}%",
                    'R²': f"{statistical_metrics.get('r_squared', np.nan):.4f}",
                    'Correlation': f"{statistical_metrics.get('correlation', np.nan):.4f}",
                    'Execution Time': f"{result.execution_time:.2f}s",
                    'Status': '✅ Success'
                })
            else:
                summary_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'RMSE': 'Failed',
                    'MAE': 'Failed',
                    'MAPE': 'Failed',
                    'R²': 'Failed',
                    'Correlation': 'Failed',
                    'Execution Time': f"{result.execution_time:.2f}s",
                    'Status': f'❌ {result.error_message[:50]}...' if result.error_message else '❌ Failed'
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Sort by RMSE (successful models first)
            successful_models = df[df['Status'] == '✅ Success'].copy()
            failed_models = df[df['Status'] != '✅ Success'].copy()
            
            if not successful_models.empty:
                # Convert RMSE to float for sorting
                successful_models['RMSE_float'] = successful_models['RMSE'].astype(float)
                successful_models = successful_models.sort_values('RMSE_float').drop('RMSE_float', axis=1)
            
            df_sorted = pd.concat([successful_models, failed_models], ignore_index=True)
            return df_sorted
        
        return pd.DataFrame()
    
    def create_rolling_validation_plot(self, rolling_results: pd.DataFrame) -> go.Figure:
        """Create rolling window validation visualization"""
        
        if rolling_results.empty:
            return go.Figure().add_annotation(text="No rolling validation data available")
        
        # Create subplots for different metrics
        metrics = ['rmse', 'mae', 'mape']
        available_metrics = [m for m in metrics if m in rolling_results.columns]
        
        if not available_metrics:
            return go.Figure().add_annotation(text="No metrics data available")
        
        fig = make_subplots(
            rows=len(available_metrics), cols=1,
            subplot_titles=[f'{m.upper()} Across Rolling Windows' for m in available_metrics],
            vertical_spacing=0.08
        )
        
        models = rolling_results['model_name'].unique()
        
        for i, metric in enumerate(available_metrics, 1):
            for model in models:
                model_data = rolling_results[rolling_results['model_name'] == model]
                successful_data = model_data[model_data['status'] == 'completed']
                
                if not successful_data.empty:
                    color = self.model_colors.get(model, f'hsl({hash(model) % 360}, 70%, 50%)')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=successful_data['window_id'],
                            y=successful_data[metric],
                            mode='lines+markers',
                            name=f'{model.replace("_", " ").title()}',
                            line=dict(color=color),
                            showlegend=(i == 1)  # Only show legend for first subplot
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            height=300 * len(available_metrics),
            title='Rolling Window Validation Results',
            template='plotly_white'
        )
        
        # Update x-axis labels
        for i in range(1, len(available_metrics) + 1):
            fig.update_xaxes(title_text='Window ID', row=i, col=1)
        
        return fig
    
    def create_model_diagnostics_plot(self, model_results: Dict[str, ModelResult]) -> go.Figure:
        """Create model diagnostics visualization"""
        
        # Collect diagnostic data
        diagnostic_data = []
        
        for model_name, result in model_results.items():
            if result.success and result.diagnostics:
                diagnostics = result.diagnostics
                
                # Extract common diagnostic metrics
                for metric_name, value in diagnostics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        diagnostic_data.append({
                            'model': model_name.replace('_', ' ').title(),
                            'metric': metric_name.upper(),
                            'value': value
                        })
        
        if not diagnostic_data:
            return go.Figure().add_annotation(text="No diagnostic data available")
        
        df_diag = pd.DataFrame(diagnostic_data)
        
        # Create separate plots for different metric types
        unique_metrics = df_diag['metric'].unique()
        
        if len(unique_metrics) == 1:
            # Single metric - simple bar chart
            fig = px.bar(df_diag, x='model', y='value', color='model',
                        title=f'Model {unique_metrics[0]} Comparison')
        else:
            # Multiple metrics - grouped bar chart
            fig = px.bar(df_diag, x='model', y='value', color='metric', barmode='group',
                        title='Model Diagnostics Comparison')
        
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_residuals_analysis(self, 
                                actual_values: pd.Series,
                                model_results: Dict[str, ModelResult]) -> go.Figure:
        """Create residuals analysis plots"""
        
        # Create subplots for each model
        n_models = sum(1 for result in model_results.values() if result.success)
        if n_models == 0:
            return go.Figure().add_annotation(text="No successful models to analyze")
        
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'{name.replace("_", " ").title()} Residuals' 
                           for name, result in model_results.items() if result.success],
            vertical_spacing=0.1
        )
        
        plot_idx = 1
        for model_name, result in model_results.items():
            if result.success and result.predictions is not None:
                # Calculate residuals
                common_idx = actual_values.index.intersection(result.predictions.index)
                if len(common_idx) > 0:
                    actual_aligned = actual_values.loc[common_idx]
                    pred_aligned = result.predictions.loc[common_idx]
                    residuals = actual_aligned - pred_aligned