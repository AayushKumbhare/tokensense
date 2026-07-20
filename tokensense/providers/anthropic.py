"""Thin LiteLLM provider config for Anthropic."""
from __future__ import annotations

PROVIDER_NAME = "anthropic"
API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-sonnet-5"

# Prompt caching (see cache/decision.py): ephemeral cache, 5-minute TTL that
# refreshes on every cache read. Requires explicit cache_control markers on
# the content blocks to cache.
SUPPORTS_PROMPT_CACHING = True
PROMPT_CACHE_TTL_SECONDS = 300
NEEDS_CACHE_MARKERS = True


def to_litellm_model(model: str) -> str:
    return model  # LiteLLM recognizes bare Anthropic model names directly
