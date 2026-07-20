"""Thin LiteLLM provider config for Gemini."""
from __future__ import annotations

PROVIDER_NAME = "gemini"
API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-flash"

# Prompt caching: implicit caching on Gemini 2.5 models, no markers required.
SUPPORTS_PROMPT_CACHING = True
PROMPT_CACHE_TTL_SECONDS = 300
NEEDS_CACHE_MARKERS = False


def to_litellm_model(model: str) -> str:
    return model if model.startswith("gemini/") else f"gemini/{model}"
