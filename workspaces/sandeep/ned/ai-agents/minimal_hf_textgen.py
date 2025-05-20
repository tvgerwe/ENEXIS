# minimal_hf_textgen.py
"""
Minimal script to run GPT-2 text generation locally using HuggingFace Transformers.
No API key or internet connection required after model download.
"""
from transformers import pipeline

PROMPT = "Write a short summary of the importance of open-source AI models."

# Load the GPT-2 model and tokenizer locally
pipe = pipeline("text-generation", model="gpt2")

# Generate text
output = pipe(PROMPT, max_new_tokens=128)
print("\nOutput:\n" + output[0]['generated_text'])
