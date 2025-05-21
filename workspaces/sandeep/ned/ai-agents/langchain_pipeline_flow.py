# langchain_pipeline_flow.py
# env - conda activate enexis-may-03-env-ai-run
# SUMMARY: Provides LLM-powered natural language querying and summarization over the AI agent pipeline flow logs. Tracks each API call (build, forecast, cross-validation) and enables conversational visibility into pipeline execution using LangChain and HuggingFaceEndpoint.

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

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
def query_pipeline_flow(question=None, log_path=None, model_name=None):
    """
    Query the pipeline flow logs using a local HuggingFace text-generation model (default: gpt2).
    No API key or internet required. Only local models are supported.
    """
    # Recommended local models (context window in tokens):
    # - gpt2: 1024
    # - bigscience/bloom-560m: 2048
    # - facebook/opt-1.3b: 2048
    # - facebook/opt-2.7b: 2048
    # - mistralai/Mistral-7B-Instruct-v0.2: 32768 (requires lots of RAM/GPU)
    # - meta-llama/Llama-2-7b-hf: 4096 (requires lots of RAM/GPU)
    # - tiiuae/falcon-7b-instruct: 2048 (requires lots of RAM/GPU)
    if question is None:
        question = "Summarize the pipeline flow."
    if log_path is None:
        log_path = LOG_PATH
    if model_name is None:
        model_name = "gpt2"  # Default to gpt2 for local inference
    logger.info(f"Querying pipeline flow with question: {question}")
    df = load_pipeline_logs(log_path)
    if df.empty:
        logger.warning("No pipeline flow logs found.")
        return "No pipeline flow logs found."
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Next step for summary")
    try:
        # Try to load the model/tokenizer locally only
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Could not load model '{model_name}' locally: {e}")
        return f"Model '{model_name}' is not available locally. Please download it first using transformers CLI or Python."
    # Only summarize the first three log lines
    summaries = []
    for idx, row in df.head(2).iterrows():
        log_text = row.to_string()
        prompt = f"Given the following pipeline log entry:\n{log_text}\n\nSummarize this log entry."
        try:
            output = pipe(prompt, max_new_tokens=128)
            if isinstance(output, list) and len(output) > 0:
                first = output[0]
                if isinstance(first, dict) and 'generated_text' in first:
                    summaries.append(first['generated_text'].strip())
                else:
                    summaries.append(str(first))
            else:
                summaries.append("Text generation returned no output.")
        except Exception as e:
            logger.error(f"Text generation failed for log {idx}: {e}")
            summaries.append(f"Text generation failed for log {idx}: {e}")
    # Combine all summaries and answer the user's question
    combined_summary = "\n".join(summaries)
    final_prompt = f"Given the following summarized pipeline logs:\n{combined_summary}\n\nAnswer this question: {question}"
    try:
        output = pipe(final_prompt, max_new_tokens=256)
        if isinstance(output, list) and len(output) > 0:
            first = output[0]
            if isinstance(first, dict) and 'generated_text' in first:
                return first['generated_text']
            return str(first)
        return "Text generation returned no output."
    except Exception as e:
        logger.error(f"Text generation failed for final summary: {e}")
        return f"Text generation failed for final summary: {e}"

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
