# flows/full_pipeline_flow.py
from prefect import flow, task
from src.data_ingestion.entsoe_load import fetch_with_retries
from src.data_ingestion.ingest_date import main as ingest_date_main
from src.data_processing.entsoe_dataprocessing import conn as entsoe_processing
from src.data_master.build_master_observed import build_master as build_observed
from src.data_master.build_master_predictions import build_master as build_predictions

@task
def ingest_and_process():
    # Run date ingestion
    ingest_date_main()
    
    # ENTSOE data is processed automatically when imported
    # Weather data needs to be run manually from notebooks for now
    print("⚠️ Run weather notebooks manually:")
    print("1. API_open_meteo_historical.ipynb")
    print("2. API_open_meteo_preds.ipynb") 
    print("3. transform_weather_obs.ipynb")
    print("4. transform_weather_preds.ipynb")

@task  
def build_masters():
    master_observed = build_observed()
    master_predictions = build_predictions()
    return master_observed, master_predictions

@flow
def full_pipeline_flow():
    ingest_and_process()
    return build_masters()

if __name__ == "__main__":
    full_pipeline_flow()