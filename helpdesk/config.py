"""
helpdesk/config.py - Configuration constants and paths for Helpdesk MAS.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TABLES_DIR = DATA_DIR / "tables"
CHROMA_DIR = DATA_DIR / "chroma"
LOGS_DIR = BASE_DIR / "logs"
DRAFTS_DIR = LOGS_DIR / "drafts"
TRACES_DIR = LOGS_DIR / "traces"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report"

# Ensure runtime directories exist
for d in [DATA_DIR, TABLES_DIR, CHROMA_DIR, LOGS_DIR, DRAFTS_DIR, TRACES_DIR, RESULTS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Environment and Ollama settings
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
os.environ["OLLAMA_API_BASE"] = OLLAMA_API_BASE
# OLLAMA_CONTEXT_LENGTH / OLLAMA_MAX_LOADED_MODELS are read by the Ollama *server*, so they must be
# set where `ollama serve` runs (see README); setting them here would have no effect.

# Models (override with environment variables, see .env.example)
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "ollama_chat/qwen3:8b")
LLAMA_CHAT_MODEL = os.getenv("LLAMA_CHAT_MODEL", "ollama_chat/llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "helpdesk_corpus"
NUM_CTX = int(os.getenv("HELPDESK_NUM_CTX", "8192"))  # context window requested per call

# RAG & Guardrail parameters
DEFAULT_TOP_K = 5
# Cosine-distance cutoff (smaller = more similar). Calibrated on probe questions with nomic-embed-text:
# in-scope questions had best distance <= 0.26, out-of-scope ones >= 0.30, but short keyword
# queries written by the agents reach ~0.32, so 0.35 is used and borderline cases are left to the LLM.
DEFAULT_TAU_THRESHOLD = float(os.getenv("HELPDESK_TAU", "0.35"))

# Escalation & unanswered log
UNANSWERED_LOG_FILE = LOGS_DIR / "unanswered.jsonl"
