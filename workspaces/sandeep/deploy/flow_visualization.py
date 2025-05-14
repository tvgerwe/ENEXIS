# flow_visualization.py
from graphviz import Digraph

def visualize_pipeline():
    dot = Digraph(comment='Data Pipeline')

    # Add nodes (tasks) to the pipeline
    dot.node('A', 'Task 1')
    dot.node('B', 'Task 2')
    dot.node('C', 'Task 3')

    # Add edges (dependencies between tasks)
    dot.edge('A', 'B')
    dot.edge('B', 'C')

    # Render the flowchart (you can specify a file name here)
    dot.render('pipeline_flow', format='png', cleanup=True)

    print("Flowchart generated and saved as pipeline_flow.png")
