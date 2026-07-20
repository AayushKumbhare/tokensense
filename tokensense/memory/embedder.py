"""Embedding model abstraction, routed through LiteLLM so any provider (or an
Ollama-served local model) can be used without a separate embedding client.

Local-first default: nomic-embed-text via Ollama, so Ollama-only users never
silently send conversation content to a hosted API. OpenAI text-embedding-3-*
models remain available and are requested at EMBEDDING_DIM dimensions (they
support truncation natively) so every supported model fits the one schema.
"""
from __future__ import annotations

import litellm

from .store import EMBEDDING_DIM

DEFAULT_EMBEDDING_MODEL = "ollama/nomic-embed-text"

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
