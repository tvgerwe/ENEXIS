# pipeline.py

import pandas as pd
from tasks import run_tasks

def load_data():
    """
    Load the data. This is just a placeholder for data loading.
    """
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [10, 20, 30, 40],
        'col3': [100, 200, 300, 400]
    })
    return data

def save_data(data):
    """
    Save the processed data to a CSV file or any other format.
    """
    data.to_csv('output.csv', index=False)
    print("Data saved to output.csv")

def run_pipeline():
    """
    Run the complete pipeline.
    """

    print("Starting pipeline...")

    # Load the data
    input_data = load_data()

    # Run tasks
    processed_data = run_tasks(input_data)

    # Save the final processed data
    save_data(processed_data)

    print("Pipeline completed!")
