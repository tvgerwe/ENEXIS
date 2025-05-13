from prefect import flow, task
from src.data_ingestion.entsoe import fetch_entsoe_data
from src.data_ingestion.weather import fetch_weather_data
from src.data_ingestion.sin_cos import generate_time_features
from src.data_ingestion.merge_sources import merge_dataframes
from src.data_processing.cleaning import clean_dataframe
from src.data_processing.validation import validate_dataframe
from src.data_processing.feature_eng import engineer_features
from src.data_processing.split import split_data

@task
def fetch_all_data():
    df_entsoe = fetch_entsoe_data()
    df_weather = fetch_weather_data()
    df_time = generate_time_features()
    return df_entsoe, df_weather, df_time

@task
def merge_and_process(df_entsoe, df_weather, df_time):
    merged_df = merge_dataframes(df_entsoe, df_weather, df_time)
    cleaned_df = clean_dataframe(merged_df)
    validate_dataframe(cleaned_df)
    final_df = engineer_features(cleaned_df)
    return final_df

@task
def split_dataset(df):
    split_data(df)

@flow(name="Electricity Price Forecasting Full Pipeline")
def full_pipeline_flow():
    df_entsoe, df_weather, df_time = fetch_all_data()
    final_df = merge_and_process(df_entsoe, df_weather, df_time)
    split_dataset(final_df)

if __name__ == "__main__":
    full_pipeline_flow()
