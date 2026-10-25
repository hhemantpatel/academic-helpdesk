"""
helpdesk/models.py - Build the ADK model object for an Ollama model served through LiteLLM.
"""

from google.adk.models.lite_llm import LiteLlm

from helpdesk.config import NUM_CTX


def make_model(model_name: str) -> LiteLlm:
    """LiteLlm wrapper with a context window large enough for instructions + tool schemas + passages
    (Ollama's per-request default is much smaller and silently truncates the prompt)."""
    name = model_name if "/" in model_name else f"ollama_chat/{model_name}"
    return LiteLlm(model=name, num_ctx=NUM_CTX)
