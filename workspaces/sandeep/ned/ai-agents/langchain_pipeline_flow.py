# langchain_pipeline_flow.py
# env - conda activate enexis-may-03-env-ai-run
# SUMMARY: Provides LLM-powered natural language querying and summarization over the AI agent pipeline flow logs. Tracks each API call (build, forecast, cross-validation) and enables conversational visibility into pipeline execution using LangChain and HuggingFaceEndpoint.

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# To use HuggingFaceEndpoint, set the HUGGINGFACEHUB_API_TOKEN environment variable or pass it as a parameter.

LOG_PATH = Path(__file__).parent / "pipeline_flow_log.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline_flow')


CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"❌ Config not found at : {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

HUGGINGFACE_API_TOKEN = config['ned']['huggingface_key']

def log_pipeline_step(step=None, input_data=None, output_data=None, log_path=LOG_PATH):
    """Log a pipeline step with input/output for later LLM analysis. Defaults to safe values if not provided."""
    if step is None:
        step = "unknown_step"
    if input_data is None:
        input_data = {}
    if output_data is None:
        output_data = {}
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "input": input_data,
        "output": output_data
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logger.info(f"Logged pipeline step: {step} | input: {input_data} | output: {output_data}")

def load_pipeline_logs(log_path=LOG_PATH):
    """Load pipeline logs as a DataFrame."""
    if not Path(log_path).exists():
        return pd.DataFrame()
    logs = [json.loads(line) for line in open(log_path)]
    return pd.DataFrame(logs)

# Example LangChain-powered query function
def query_pipeline_flow(question=None, log_path=None, huggingfacehub_api_token=None, model_name=None):
    """
    Minimal: Query the pipeline flow logs using a supported HuggingFace text-generation model (default: tiiuae/falcon-7b-instruct).
    Uses direct requests to the HuggingFace Inference API for maximum compatibility.
    """
    import requests
    if question is None:
        question = "Summarize the pipeline flow."
    if log_path is None:
        log_path = LOG_PATH
    if huggingfacehub_api_token is None:
        huggingfacehub_api_token = HUGGINGFACE_API_TOKEN
    if model_name is None:
        model_name = "tiiuae/falcon-7b-instruct"
    logger.info(f"Querying pipeline flow with question: {question}")
    df = load_pipeline_logs(log_path)
    if df.empty:
        logger.warning("No pipeline flow logs found.")
        return "No pipeline flow logs found."
    logs_text = df.to_string(index=False)
    prompt = f"Given the following pipeline logs:\n{logs_text}\n\nAnswer this question: {question}"
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {huggingfacehub_api_token}"}
    payload = {"inputs": prompt}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        logger.error(f"HuggingFace API error: {response.status_code} {response.text}")
        return f"HuggingFace API error: {response.status_code} {response.text}"
    result = response.json()
    if isinstance(result, list) and 'generated_text' in result[0]:
        return result[0]['generated_text']
    return str(result)

# Example usage (for testing)
if __name__ == "__main__":
    # Log a sample build step
    log_pipeline_step(
        step="build_model",
        input_data={"csv_path": "temp_validate.csv", "train_start": "2025-01-01"},
        output_data={"success": True, "model_path": "prophet-ai-agent-model.pkl"}
    )
    # Query the flow
    print(query_pipeline_flow("What steps were executed in the last run?", model_name="gpt2"))
