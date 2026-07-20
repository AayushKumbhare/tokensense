from datetime import datetime, timedelta, timezone

from tokensense.cache.decision import ContextStrategy, choose_strategy


def test_uses_native_cache_within_ttl():
    future = datetime.now(timezone.utc) + timedelta(minutes=2)
    assert (
        choose_strategy(provider_ttl_expires_at=future, supports_prompt_caching=True)
        == ContextStrategy.NATIVE_CACHE
    )


def test_falls_back_to_rag_after_ttl_expires():
    past = datetime.now(timezone.utc) - timedelta(minutes=2)
    assert (
        choose_strategy(provider_ttl_expires_at=past, supports_prompt_caching=True)
        == ContextStrategy.RAG_RETRIEVAL
    )


def test_falls_back_to_rag_when_provider_unsupported():
    future = datetime.now(timezone.utc) + timedelta(minutes=2)
    assert (
        choose_strategy(provider_ttl_expires_at=future, supports_prompt_caching=False)
        == ContextStrategy.RAG_RETRIEVAL
    )


def test_falls_back_to_rag_when_never_cached():
    assert (
        choose_strategy(provider_ttl_expires_at=None, supports_prompt_caching=True)
        == ContextStrategy.RAG_RETRIEVAL
    )
