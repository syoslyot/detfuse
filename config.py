import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_URL: str     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL: str   = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
