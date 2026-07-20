"""Thin LiteLLM provider config for OpenAI."""
from __future__ import annotations

PROVIDER_NAME = "openai"
API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-4o-mini"

# Prompt caching: automatic prefix caching on 1024+ token prompts, no markers.
# OpenAI documents ~5–10 min retention; 300s is the conservative bound.
SUPPORTS_PROMPT_CACHING = True
PROMPT_CACHE_TTL_SECONDS = 300
NEEDS_CACHE_MARKERS = False


def to_litellm_model(model: str) -> str:
    return model  # LiteLLM recognizes bare OpenAI model names directly
