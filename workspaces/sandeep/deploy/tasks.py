# tasks.py
from agentic_ai import task_1, task_2, task_3
from rich.console import Console
from rich.progress import track

def run_tasks(input_data):
    """
    Execute the tasks sequentially.
    """

    for _ in track(range(1), description="Executing Tasks..."):
        # Task 1
        result_1 = task_1(input_data)

        # Task 2
        result_2 = task_2(result_1)

        # Task 3
        result_3 = task_3(result_2)

    return result_3
