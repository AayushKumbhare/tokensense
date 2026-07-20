"""Thin LiteLLM provider config for a local Ollama model."""
from __future__ import annotations

PROVIDER_NAME = "ollama"
API_KEY_ENV = None  # local — no API key required
DEFAULT_MODEL = "llama3.2"

# No provider-side prompt cache — documents always go through RAG retrieval.
SUPPORTS_PROMPT_CACHING = False
PROMPT_CACHE_TTL_SECONDS = 0
NEEDS_CACHE_MARKERS = False


def to_litellm_model(model: str) -> str:
    return model if model.startswith("ollama/") else f"ollama/{model}"
