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

**prophet_build_model_core.py**  
SUMMARY: Core logic for Prophet model build and training. Provides a function for model training, hyperparameter tuning, and artifact saving. Used by the orchestration and API layers for modular time series workflows.

---

**prophet_forecast_core.py**  
SUMMARY: Core logic for Prophet-based forecasting using saved models. Provides a function for generating forecasts from trained Prophet models, handling regressor management and robust time range logic. Used by the orchestration and API layers for modular time series forecasting workflows.

---

**prophet_crossvalidation_core.py**  
SUMMARY: Core logic and FastAPI microservice for Prophet cross-validation and performance metrics. Provides a function for running Prophet's cross-validation and metrics, handling file uploads, column normalization, and robust error/debug logging. Used by the unified API and orchestration layer for modular time series validation workflows.

---

**prophet_api_client.py**  
SUMMARY: Standalone Python client for calling the Prophet model build API endpoint. Useful for testing, scripting, or integrating Prophet model training into other Python workflows.

---

**prophet_api_forecast_client.py**  
SUMMARY: Standalone Python client for calling the Prophet forecast API endpoint. Useful for testing, scripting, or integrating Prophet-based forecasting into other Python workflows.

---

**langchain_pipeline_flow.py**  
SUMMARY: Provides LLM-powered natural language querying and summarization over the AI agent pipeline flow logs. Tracks each API call (build, forecast, cross-validation) and enables conversational visibility into pipeline execution using LangChain. Integrates with the orchestration layer to log each step for later analysis or Q&A.
