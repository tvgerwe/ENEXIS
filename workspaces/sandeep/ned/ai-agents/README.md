# ai-agents: Modular Prophet Pipeline

This folder contains a modular, API-driven pipeline for Prophet time series modeling and forecasting, designed for robust orchestration, extensibility, and research reproducibility.

## File Summaries

---

**agent_api_calls.py**  
SUMMARY: Python client library for programmatically calling the FastAPI endpoints in main_api.py. Provides robust, logged functions for model build, forecast, and cross-validation. Used by orchestration scripts and CLI/scheduler for API-driven workflows.

---

**agent_orchestration.py**  
SUMMARY: Orchestration and business logic layer for the modular Prophet pipeline. Provides functions for model build, forecast, and cross-validation by calling the core logic directly. Used by main_api.py (API server) and main.py (scheduler/CLI) for unified, non-HTTP orchestration.

---

**main_api.py**  
SUMMARY: Unified FastAPI server for the modular Prophet pipeline. Exposes endpoints for model build, forecast, and cross-validation, delegating to agent_orchestration.py for business logic. Supports file upload, robust logging, and error handling. Entry point for API-driven workflows and integration with external systems.

---

**main.py**  
SUMMARY: Scheduler and CLI entry point for orchestrating the modular Prophet pipeline. Uses agent_api_calls.py to call the unified API endpoints for model build, forecast, and cross-validation. Supports scheduled and manual runs for automated time series workflows.

---

**prophet_api.py**  
SUMMARY: FastAPI microservice and core logic for Prophet model training and evaluation. Provides a core function for model build, hyperparameter tuning, and metrics, as well as an API endpoint for file upload and orchestration. Used by the unified API and orchestration layer for modular time series model building workflows.

---

**prophet_api_forecast.py**  
SUMMARY: FastAPI microservice and core logic for Prophet-based forecasting using saved models. Provides a core function for generating forecasts from trained Prophet models, handling file uploads, regressor management, and robust time range logic. Used by the unified API and orchestration layer for modular time series forecasting workflows.

---

**prophet_crossvalidation_core.py**  
SUMMARY: FastAPI microservice and core logic for Prophet cross-validation and performance metrics. Provides a core function for running Prophet's cross-validation and metrics, handling file uploads, column normalization, and robust error/debug logging. Used by the unified API and orchestration layer for modular time series validation workflows.
