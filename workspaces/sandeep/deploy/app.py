import gradio as gr
from agentic_ai import agentic_ai

def run_agent(prompt):
    return agentic_ai(prompt)

gr.Interface(fn=run_agent, inputs="text", outputs="text", title="Agentic AI Prompt Runner").launch()
