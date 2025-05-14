# streamlit_pipeline.py
import streamlit as st

def show_pipeline_visualization():
    st.title('Pipeline Flow Visualization')

    # Simple pipeline overview
    st.write("""
    **Task 1** -> **Task 2** -> **Task 3**
    """)

    # Visualization (can use any graphics tool here, such as a chart)
    st.line_chart([1, 2, 3])  # Example of a simple chart

if __name__ == "__main__":
    show_pipeline_visualization()
