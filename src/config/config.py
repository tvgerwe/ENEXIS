#!/usr/bin/env python3

import os
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('config')

# Determine project root - will work regardless of where the repo is cloned
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Assuming config.py is in src/config/

# Default configuration
DEFAULT_CONFIG = {
    "api": {
        "ned": {
            "endpoint": "https://api.ned.nl/v1/utilizations",
            "api_key": "21702b116e4c72974d62853623de0adcb0f530d98591b308a41a881735267bbb",
            "types": [2]
        },
        "entsoe": {
            "api_key": "82aa28d4-59f3-4e3a-b144-6659aa9415b5",
            "country": "NL", 
            "neighbors": ["BE", "DE", "GB", "DK", "NO"],
            "default_start": "2025-01-01T00:00:00Z"
        },
        "open_meteo": {
            "coordinates": {
                "latitude": 52.12949,
                "longitude": 5.20514
            },
            "default_start": "2025-01-01"
        }
    },
    "database": {
        "main_db": "WARP.db",
        "logs_db": "logs.db",
        "data_dir": "data"
    },
    "ml_settings": {
        "time_splits": {
            "train_start": "2025-01-01",
            "train_end": "2025-04-15",
            "validation_start": "2025-04-16",
            "validation_end": "2025-04-30",
            "test_start": "2025-05-01",
            "test_end": "2025-05-11"
        }
    }
}

def get_config():
    """
    Load configuration with fallbacks:
    1. Try config.json in project root
    2. Try config.json in src/config/
    3. Use default config with any environment variables applied
    
    Returns a configuration dictionary
    """
    config_paths = [
        PROJECT_ROOT / "config.json",
        PROJECT_ROOT / "src" / "config" / "config.json",
        PROJECT_ROOT / "config" / "config.json"
    ]
    
    config = DEFAULT_CONFIG
    
    # Try to load from JSON file
    for path in config_paths:
        if path.exists():
            logger.info(f"Loading configuration from {path}")
            try:
                with open(path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update default config with loaded values
                    _deep_update(config, loaded_config)
                break
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
    
    # Override with environment variables
    # Example: ENEXIS_API_NED_KEY will override config["api"]["ned"]["api_key"]
    for env_var, env_value in os.environ.items():
        if env_var.startswith("ENEXIS_"):
            parts = env_var.lower().split('_')[1:]  # Remove ENEXIS_ prefix
            
            # Navigate the config dictionary
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1]] = env_value
    
    return config

def _deep_update(original, update):
    """
    Recursively update a dictionary with another dictionary
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            _deep_update(original[key], value)
        else:
            original[key] = value

def get_db_path(db_name=None):
    """
    Get absolute path to database
    """
    config = get_config()
    
    # Use specified db_name or default to main_db
    db_name = db_name or config["database"]["main_db"]
    
    # Build data directory path relative to project root
    data_dir = PROJECT_ROOT / config["database"]["data_dir"]
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir / db_name

def get_api_key(service):
    """Get API key for a specific service"""
    config = get_config()
    return config["api"][service]["api_key"]

# Create the module variables for easy importing
config = get_config()
MAIN_DB_PATH = get_db_path()
LOGS_DB_PATH = get_db_path("logs.db")
