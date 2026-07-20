"""Decide whether a document's provider-side prompt cache is still worth using,
or whether RAG retrieval over its chunks is the cheaper path (see project doc:
Static Content Strategy: Provider Caching vs. RAG)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum


class ContextStrategy(str, Enum):
    NATIVE_CACHE = "native_cache"
    RAG_RETRIEVAL = "rag_retrieval"


def choose_strategy(
    *, provider_ttl_expires_at: datetime | None, supports_prompt_caching: bool
) -> ContextStrategy:
    """Within the provider's TTL window, reuse the native cache (cheap reads).
    Once it's expired — or the provider doesn't support caching at all — fall
    back to retrieving the relevant chunk instead of resending the full document."""
    if not supports_prompt_caching or provider_ttl_expires_at is None:
        return ContextStrategy.RAG_RETRIEVAL
    if provider_ttl_expires_at > datetime.now(timezone.utc):
        return ContextStrategy.NATIVE_CACHE
    return ContextStrategy.RAG_RETRIEVAL
