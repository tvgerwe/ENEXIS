# langchain_pipeline_flow.py
# SUMMARY: Provides LLM-powered natural language querying and summarization over the AI agent pipeline flow logs. Tracks each API call (build, forecast, cross-validation) and enables conversational visibility into pipeline execution using LangChain.

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Optional: Uncomment and configure your LLM provider
# from langchain.llms import HuggingFaceHub
# from langchain.agents import create_pandas_dataframe_agent
# To use HuggingFaceHub, set the HUGGINGFACEHUB_API_TOKEN environment variable or pass it as a parameter.

LOG_PATH = Path(__file__).parent / "pipeline_flow_log.jsonl"

def log_pipeline_step(step, input_data, output_data, log_path=LOG_PATH):
    """Log a pipeline step with input/output for later LLM analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "input": input_data,
        "output": output_data
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def load_pipeline_logs(log_path=LOG_PATH):
    """Load pipeline logs as a DataFrame."""
    if not Path(log_path).exists():
        return pd.DataFrame()
    logs = [json.loads(line) for line in open(log_path)]
    return pd.DataFrame(logs)

# Example LangChain-powered query function (requires HuggingFaceHub API token)
def query_pipeline_flow(question, log_path=LOG_PATH, huggingfacehub_api_token=None, model_name="google/flan-t5-base"):
    """Query the pipeline flow logs using an LLM via LangChain (HuggingFaceHub)."""
    df = load_pipeline_logs(log_path)
    if df.empty:
        return "No pipeline flow logs found."
    # Use HuggingFaceHub LLM (uncomment below to enable)
    from langchain.llms import HuggingFaceHub
    from langchain.agents import create_pandas_dataframe_agent
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, repo_id=model_name)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent.run(question)
    return f"[LangChain LLM query would run here using HuggingFaceHub model '{model_name}']"

# Example usage (for testing)
if __name__ == "__main__":
    # Log a sample build step
    log_pipeline_step(
        step="build_model",
        input_data={"csv_path": "data.csv", "train_start": "2025-01-01"},
        output_data={"success": True, "model_path": "model.pkl"}
    )
    # Query the flow
    print(query_pipeline_flow("What steps were executed in the last run?"))
