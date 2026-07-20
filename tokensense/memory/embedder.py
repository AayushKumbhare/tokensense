"""Embedding model abstraction, routed through LiteLLM so any provider (or an
Ollama-served local model) can be used without a separate embedding client.

Default: OpenAI text-embedding-3-small, requested at EMBEDDING_DIM dimensions
(the model supports truncation natively) — one API key covers both this and
the default summarizer, which is the whole point (see docs/decisions.md #8).
Ollama's nomic-embed-text remains available and fits the same schema for
users who'd rather keep everything local and avoid API costs.
"""
from __future__ import annotations

import litellm

from .store import EMBEDDING_DIM

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Models that accept a `dimensions` request parameter and therefore can be
# asked to match EMBEDDING_DIM directly.
_RESIZABLE_PREFIXES = ("text-embedding-3",)


class Embedder:
    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL):
        self.model = model
        self._kwargs = (
            {"dimensions": EMBEDDING_DIM} if model.startswith(_RESIZABLE_PREFIXES) else {}
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = litellm.embedding(model=self.model, input=texts, **self._kwargs)
        return [item["embedding"] for item in response["data"]]
