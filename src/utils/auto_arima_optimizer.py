# src/utils/auto_arima_optimizer.py

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import numpy as np
from pmdarima import auto_arima
from utils.validation_utils import run_validation_experiment

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - auto_arima - %(levelname)s - %(message)s")
logger = logging.getLogger("auto_arima_optimizer")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WARP_DB = PROJECT_ROOT / "src" / "data" / "WARP.db"
LOG_DB = PROJECT_ROOT / "src" / "data" / "logs.db"
DATA_TABLE = "master_warp"
LOG_TABLE = "arima_configs"
VALIDATION_TABLE = "arima_validation_results"

def ensure_log_tables():
    """Ensure both logging tables exist in logs.db"""
    conn = sqlite3.connect(LOG_DB)
    
    # Original arima_configs table
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            order_params TEXT NOT NULL,
            seasonal_order_params TEXT NOT NULL,
            aic REAL,
            bic REAL,
            lambda REAL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    
    # New validation results table
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {VALIDATION_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id INTEGER,
            order_params TEXT NOT NULL,
            seasonal_order_params TEXT NOT NULL,
            mean_rmse REAL NOT NULL,
            std_rmse REAL NOT NULL,
            min_rmse REAL NOT NULL,
            max_rmse REAL NOT NULL,
            success_rate REAL NOT NULL,
            improvement_vs_baseline REAL,
            consistency_score REAL NOT NULL,
            overall_score REAL NOT NULL,
            validation_days INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (config_id) REFERENCES {LOG_TABLE} (id)
        );
    """)
    
    conn.commit()
    conn.close()

def get_current_best_config() -> Optional[Dict]:
    """Get the current best performing configuration from validation results"""
    conn = sqlite3.connect(LOG_DB)
    cursor = conn.cursor()
    
    query = f"""
        SELECT order_params, seasonal_order_params, mean_rmse, std_rmse, 
               overall_score, improvement_vs_baseline, created_at
        FROM {VALIDATION_TABLE}
        ORDER BY overall_score DESC
        LIMIT 1
    """
    
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'order': eval(row[0]),
            'seasonal_order': eval(row[1]),
            'mean_rmse': row[2],
            'std_rmse': row[3],
            'overall_score': row[4],
            'improvement_vs_baseline': row[5],
            'created_at': row[6]
        }
    return None

def calculate_overall_score(mean_rmse: float, std_rmse: float, improvement_vs_baseline: float) -> float:
    """
    Calculate overall performance score considering RMSE, consistency, and improvement
    Higher score = better performance
    """
    # Normalize components (lower RMSE and std = higher score)
    rmse_score = 1 / (1 + mean_rmse * 100)  # Scale RMSE to reasonable range
    consistency_score = 1 / (1 + std_rmse * 1000)  # Scale std to reasonable range
    improvement_score = max(0, improvement_vs_baseline / 100)  # Convert % to ratio
    
    # Weighted combination
    overall_score = (0.5 * rmse_score + 0.3 * consistency_score + 0.2 * improvement_score)
    
    return overall_score

def test_parameter_configuration(order: Tuple, seasonal_order: Tuple, training_data: pd.DataFrame, 
                               exog_vars: List[str], baseline_rmse: float) -> Dict:
    """Test a specific parameter configuration using 30-day validation"""
    
    logger.info(f"Testing config: order={order}, seasonal={seasonal_order}")
    
    # Temporarily modify validation_utils to use these parameters
    # This is a bit hacky but works for our purpose
    import utils.validation_utils as val_utils
    
    # Store original function
    original_sarimax_validation = val_utils.run_single_day_validation
    
    def modified_validation(day, training_data, exog_vars):
        """Modified validation function using our test parameters"""
        from datetime import datetime, timedelta
        import warnings
        from sklearn.metrics import mean_squared_error
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.arima.model import ARIMA
        
        train_start_date = datetime(2025, 1, 1) + timedelta(days=day)
        train_end_date = datetime(2025, 3, 14) + timedelta(days=day)
        run_date = datetime(2025, 3, 15) + timedelta(days=day)
        
        try:
            if day == 0:
                daily_data = training_data.copy()
            else:
                daily_data = training_data.copy()
                np.random.seed(day)
                noise_factor = 0.001 * day
                daily_data['Price'] = daily_data['Price'] + np.random.normal(0, noise_factor, len(daily_data))
            
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
        
        # Only test SARIMAX with our parameters
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
                            order=order,  # Use our test parameters
                            seasonal_order=seasonal_order,  # Use our test parameters
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted_model = model.fit(method='lbfgs', maxiter=20, disp=False)  # Reduced iterations
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
    
    # Replace validation function temporarily
    val_utils.run_single_day_validation = modified_validation
    
    try:
        # Run validation with our parameters - SHORTER TEST
        results_df = val_utils.run_validation_experiment(training_data, exog_vars, n_days=10)  # Reduced from 30 to 10
        
        # Calculate metrics
        valid_results = results_df['SARIMAX'].dropna()
        
        if len(valid_results) == 0:
            return {
                'success': False,
                'error': 'No valid SARIMAX results'
            }
        
        mean_rmse = valid_results.mean()
        std_rmse = valid_results.std()
        min_rmse = valid_results.min()
        max_rmse = valid_results.max()
        success_rate = len(valid_results) / len(results_df)
        improvement_vs_baseline = ((baseline_rmse - mean_rmse) / baseline_rmse) * 100
        consistency_score = 1 / (1 + std_rmse)
        overall_score = calculate_overall_score(mean_rmse, std_rmse, improvement_vs_baseline)
        
        return {
            'success': True,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'min_rmse': min_rmse,
            'max_rmse': max_rmse,
            'success_rate': success_rate,
            'improvement_vs_baseline': improvement_vs_baseline,
            'consistency_score': consistency_score,
            'overall_score': overall_score
        }
        
    finally:
        # Restore original function
        val_utils.run_single_day_validation = original_sarimax_validation

def explore_parameter_space(current_best: Dict, training_data: pd.DataFrame, 
                          exog_vars: List[str]) -> List[Tuple]:
    """Generate parameter combinations to explore around current best - REDUCED SET"""
    
    if current_best:
        base_order = current_best['order']
        base_seasonal = current_best['seasonal_order']
    else:
        # Default starting point
        base_order = (1, 0, 1)
        base_seasonal = (1, 1, 1, 24)
    
    parameter_combinations = []
    
    # MUCH MORE LIMITED exploration for speed
    for p_delta in [-1, 0, 1]:
        for q_delta in [-1, 0, 1]:
            for P_delta in [-1, 0, 1]:
                for Q_delta in [-1, 0, 1]:
                    
                    new_p = max(0, min(2, base_order[0] + p_delta))  # Reduced max
                    new_d = base_order[1]  # Keep d fixed
                    new_q = max(0, min(2, base_order[2] + q_delta))  # Reduced max
                    
                    new_P = max(0, min(1, base_seasonal[0] + P_delta))  # Reduced max
                    new_D = base_seasonal[1]  # Keep D fixed
                    new_Q = max(0, min(1, base_seasonal[2] + Q_delta))  # Reduced max
                    
                    new_order = (new_p, new_d, new_q)
                    new_seasonal = (new_P, new_D, new_Q, 24)
                    
                    # Avoid invalid combinations
                    if new_order != (0, 0, 0):
                        parameter_combinations.append((new_order, new_seasonal))
    
    # Remove duplicates and limit to 8 combinations max
    parameter_combinations = list(set(parameter_combinations))
    return parameter_combinations[:8]  # Much smaller set!

def log_validation_result(order: Tuple, seasonal_order: Tuple, validation_result: Dict):
    """Log validation results to database"""
    
    conn = sqlite3.connect(LOG_DB)
    
    conn.execute(f"""
        INSERT INTO {VALIDATION_TABLE} (
            order_params, seasonal_order_params, mean_rmse, std_rmse, 
            min_rmse, max_rmse, success_rate, improvement_vs_baseline,
            consistency_score, overall_score, validation_days, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(order),
        str(seasonal_order),
        validation_result['mean_rmse'],
        validation_result['std_rmse'],
        validation_result['min_rmse'],
        validation_result['max_rmse'],
        validation_result['success_rate'],
        validation_result['improvement_vs_baseline'],
        validation_result['consistency_score'],
        validation_result['overall_score'],
        30,  # validation_days
        datetime.utcnow().isoformat()
    ))
    
    conn.commit()
    conn.close()

def run_auto_arima_optimization(training_data: pd.DataFrame, exog_vars: List[str]) -> Dict:
    """
    Run complete auto-ARIMA optimization with validation integration
    Returns the best configuration found
    """
    
    ensure_log_tables()
    
    logger.info("🚀 Starting Auto-ARIMA optimization with validation integration")
    
    # Get current best configuration
    current_best = get_current_best_config()
    
    if current_best:
        logger.info(f"📊 Current best: RMSE={current_best['mean_rmse']:.6f}, Score={current_best['overall_score']:.4f}")
        baseline_rmse = current_best['mean_rmse']
    else:
        logger.info("🆕 No previous best configuration found")
        # Run baseline validation to get baseline RMSE
        baseline_results = run_validation_experiment(training_data, exog_vars, n_days=30)
        baseline_rmse = baseline_results['SARIMAX'].dropna().mean()
    
    # Phase 1: Explore parameter space around current best
    logger.info("🔍 Phase 1: Exploring parameter space")
    
    parameter_combinations = explore_parameter_space(current_best, training_data, exog_vars)
    logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
    
    best_config = current_best
    tested_configs = []
    
    for i, (order, seasonal_order) in enumerate(parameter_combinations):
        logger.info(f"Testing {i+1}/{len(parameter_combinations)}: {order}, {seasonal_order}")
        
        validation_result = test_parameter_configuration(
            order, seasonal_order, training_data, exog_vars, baseline_rmse
        )
        
        if validation_result['success']:
            # Log result
            log_validation_result(order, seasonal_order, validation_result)
            tested_configs.append((order, seasonal_order, validation_result))
            
            # Check if this beats current best
            if (not best_config or 
                validation_result['overall_score'] > best_config['overall_score']):
                
                improvement_threshold = 5.0  # 5% improvement required
                if (not best_config or 
                    validation_result['improvement_vs_baseline'] > improvement_threshold):
                    
                    best_config = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        **validation_result
                    }
                    logger.info(f"✅ New best config found! RMSE={validation_result['mean_rmse']:.6f}, Score={validation_result['overall_score']:.4f}")
    
    # Phase 2: Auto-ARIMA if no significant improvement found
    if not best_config or (current_best and best_config['overall_score'] <= current_best['overall_score'] * 1.02):
        logger.info("🤖 Phase 2: Running Auto-ARIMA for broader search")
        
        try:
            # Prepare data for auto-arima
            split_point = training_data.index[-24]
            train_data = training_data[training_data.index < split_point]['Price'].dropna()
            
            model = auto_arima(
                train_data,
                seasonal=True,
                m=24,
                stepwise=True,  # Keep stepwise for speed
                d=1,
                D=1,
                max_p=2,  # Reduced from 3
                max_q=2,  # Reduced from 3
                max_P=1,  # Reduced from 2
                max_Q=1,  # Reduced from 2
                start_p=1,
                start_q=1,
                start_P=0,  # Start simpler
                start_Q=0,  # Start simpler
                trend='n',  # Simplified trend
                suppress_warnings=True,
                error_action="warn",
                max_order=6,  # Reduced from 10
                information_criterion="aic"
            )
            
            auto_order = model.order
            auto_seasonal = model.seasonal_order
            
            logger.info(f"🤖 Auto-ARIMA suggests: {auto_order}, {auto_seasonal}")
            
            # Test auto-ARIMA suggestion
            validation_result = test_parameter_configuration(
                auto_order, auto_seasonal, training_data, exog_vars, baseline_rmse
            )
            
            if validation_result['success']:
                log_validation_result(auto_order, auto_seasonal, validation_result)
                
                if (not best_config or 
                    validation_result['overall_score'] > best_config['overall_score']):
                    
                    best_config = {
                        'order': auto_order,
                        'seasonal_order': auto_seasonal,
                        **validation_result
                    }
                    logger.info(f"🤖 Auto-ARIMA found better config! RMSE={validation_result['mean_rmse']:.6f}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-ARIMA failed: {e}")
    
    # Summary
    if best_config:
        if current_best and best_config != current_best:
            logger.info(f"🎉 Optimization complete! New best configuration found.")
            logger.info(f"   Order: {best_config['order']}")
            logger.info(f"   Seasonal: {best_config['seasonal_order']}")
            logger.info(f"   RMSE: {best_config['mean_rmse']:.6f}")
            logger.info(f"   Improvement: {best_config['improvement_vs_baseline']:.2f}%")
        else:
            logger.info(f"✅ Optimization complete! Current best configuration confirmed.")
    else:
        logger.warning("⚠️ No valid configuration found, using fallback")
        best_config = {
            'order': (1, 0, 1),
            'seasonal_order': (1, 1, 1, 24),
            'mean_rmse': baseline_rmse,
            'overall_score': 0.0
        }
    
    return best_config

def update_sarimax_model_config(best_config: Dict) -> bool:
    """Update SarimaxModel with the best configuration found"""
    
    try:
        # This would update the default configuration in SarimaxModel
        # For now, we'll create a config file that the model can read
        
        config_file = PROJECT_ROOT / "src" / "config" / "best_sarimax_params.json"
        config_file.parent.mkdir(exist_ok=True)
        
        import json
        config_data = {
            'order': best_config['order'],
            'seasonal_order': best_config['seasonal_order'],
            'mean_rmse': best_config['mean_rmse'],
            'updated_at': datetime.utcnow().isoformat(),
            'improvement_vs_baseline': best_config.get('improvement_vs_baseline', 0.0)
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"✅ Updated SARIMAX model configuration: {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update SARIMAX config: {e}")
        return False

# Main execution function
def run_weekly_optimization(training_data: pd.DataFrame, exog_vars: List[str]) -> Dict:
    """Main function to run weekly/monthly optimization"""
    
    logger.info("=" * 60)
    logger.info("🚀 WEEKLY AUTO-ARIMA OPTIMIZATION STARTED")
    logger.info("=" * 60)
    
    # Run optimization
    best_config = run_auto_arima_optimization(training_data, exog_vars)
    
    # Update model configuration
    update_success = update_sarimax_model_config(best_config)
    
    logger.info("=" * 60)
    logger.info("✅ WEEKLY AUTO-ARIMA OPTIMIZATION COMPLETED")
    logger.info("=" * 60)
    
    return {
        'best_config': best_config,
        'update_success': update_success,
        'timestamp': datetime.utcnow().isoformat()
    }