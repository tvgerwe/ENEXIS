# agentic_ai.py
import logging
from datetime import datetime

# Setup logger
logging.basicConfig(level=logging.INFO)

def print_step_info(step_name):
    logging.info(f"[{datetime.now()}] Running step: {step_name}")

# Similarly, for other tasks, log the start time
def task_1(data):
    print_step_info("Task 1")
    data['col1'] = data['col1'] * 2
    return data

def task_2(data):
    print_step_info("Task 2")
    data['col2'] = data['col2'] + 10
    return data

def task_3(data):
    print_step_info("Task 3")
    data['col3'] = data['col3'] / 2
    return data
