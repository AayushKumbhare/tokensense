"""Cache decision layer wired into Project's document path (TASKS item 3).

Covers the strategy lifecycle end-to-end at the unit level: session-added
documents ride verbatim and trigger the cache write, live-TTL documents keep
riding (NATIVE_CACHE), expired ones fall back to RAG chunks, and non-caching
providers never send documents verbatim.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.fakes import FakeEmbedder, FakeStore, FakeSummarizer
from tokensense.memory.documents import DocumentIngestor
from tokensense.memory.retriever import Retriever
from tokensense.project import Project
from tokensense.providers import anthropic, ollama, openai
from tokensense.tracker import Tracker

import tokensense.project as project_module


@pytest.fixture
def fake_llm(monkeypatch):
    payloads: list[list[dict]] = []

    def fake_completion(model=None, messages=None, api_key=None, **kwargs):
        payloads.append(messages)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(project_module.litellm, "completion", fake_completion)
    monkeypatch.setattr(
        project_module.litellm,
        "token_counter",
        lambda model=None, messages=None: sum(len(str(m.get("content", ""))) for m in messages),
    )
    return payloads


def make_project(store: FakeStore, provider_config, name: str = "cache-test") -> Project:
    embedder = FakeEmbedder()
    return Project(
        name,
        store=store,
        retriever=Retriever(store, embedder, top_k=5),
        summarizer=FakeSummarizer(),
        chat_model="test-model",
        api_key="key",
        window_size=5,
        tracker=Tracker(),
        provider_config=provider_config,
        document_ingestor=DocumentIngestor(store, embedder),
    )


def write_spec(tmp_path, text="POST /login returns a JWT pair"):
    spec = tmp_path / "api_spec.md"
    spec.write_text(text)
    return spec


def doc_messages(payload):
    return [m for m in payload if m["role"] == "system" and "Project document" in str(m["content"])]


def rag_context(payload):
    return [
        m["content"]
        for m in payload
        if m["role"] == "system" and "Relevant context from past sessions" in str(m["content"])
    ]


def test_session_added_document_rides_verbatim_and_writes_cache(fake_llm, tmp_path):
    store = FakeStore()
    project = make_project(store, anthropic)
    doc_id = project.add_document(str(write_spec(tmp_path)))

    project.chat(messages=[{"role": "user", "content": "walk me through the login endpoint"}])

    docs = doc_messages(fake_llm[-1])
    assert len(docs) == 1
    # Anthropic needs explicit markers: content-block form with cache_control.
    block = docs[0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert "POST /login" in block["text"]
    # Payload prefix stability: the document leads the payload.
    assert fake_llm[-1][0] is docs[0]

    doc = store.documents[0]
    assert doc.id == doc_id
    assert doc.provider_ttl_expires_at is not None
    assert doc.provider_ttl_expires_at > datetime.now(timezone.utc)
    # Verbatim documents are excluded from the RAG block — no double sending.
    assert not any("POST /login" in c for c in rag_context(fake_llm[-1]))


def test_live_ttl_document_keeps_riding_in_next_session(fake_llm, tmp_path):
    store = FakeStore()
    first = make_project(store, anthropic)
    first.add_document(str(write_spec(tmp_path)))
    first.chat(messages=[{"role": "user", "content": "login endpoint?"}])
    ttl_after_first = store.documents[0].provider_ttl_expires_at

    # Fresh session (session-added set is empty) but the TTL is still live.
    second = make_project(store, anthropic)
    second.chat(messages=[{"role": "user", "content": "remind me about auth"}])

    assert len(doc_messages(fake_llm[-1])) == 1
    # Reads refresh the TTL (Anthropic semantics).
    assert store.documents[0].provider_ttl_expires_at >= ttl_after_first


def test_expired_ttl_falls_back_to_rag_chunks(fake_llm, tmp_path):
    store = FakeStore()
    first = make_project(store, anthropic)
    first.add_document(str(write_spec(tmp_path)))
    first.chat(messages=[{"role": "user", "content": "login endpoint?"}])
    store.documents[0].provider_ttl_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)

    second = make_project(store, anthropic)
    second.chat(messages=[{"role": "user", "content": "what does POST /login return?"}])

    payload = fake_llm[-1]
    assert doc_messages(payload) == []
    contexts = rag_context(payload)
    assert len(contexts) == 1 and "POST /login" in contexts[0]
    # RAG usage stamps last_used_at but must not refresh the provider TTL.
    doc = store.documents[0]
    assert doc.last_used_at is not None
    assert doc.provider_ttl_expires_at < datetime.now(timezone.utc)


def test_non_caching_provider_always_uses_rag(fake_llm, tmp_path):
    store = FakeStore()
    project = make_project(store, ollama)
    project.add_document(str(write_spec(tmp_path)))

    project.chat(messages=[{"role": "user", "content": "what does POST /login return?"}])

    payload = fake_llm[-1]
    assert doc_messages(payload) == []
    assert any("POST /login" in c for c in rag_context(payload))
    assert store.documents[0].provider_ttl_expires_at is None


def test_marker_free_provider_gets_plain_text_documents(fake_llm, tmp_path):
    store = FakeStore()
    project = make_project(store, openai)
    project.add_document(str(write_spec(tmp_path)))

    project.chat(messages=[{"role": "user", "content": "login endpoint?"}])

    docs = doc_messages(fake_llm[-1])
    assert len(docs) == 1
    # OpenAI caches prefixes automatically — plain string content, no blocks.
    assert isinstance(docs[0]["content"], str)
    assert store.documents[0].provider_ttl_expires_at is not None


def test_end_session_unpins_session_added_documents(fake_llm, tmp_path):
    store = FakeStore()
    project = make_project(store, anthropic)
    project.add_document(str(write_spec(tmp_path)))
    project.chat(messages=[{"role": "user", "content": "login endpoint?"}])
    project.end_session()
    store.documents[0].provider_ttl_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)

    # Same Project object, new sub-conversation: the pin must not survive the
    # session boundary, so with an expired TTL the doc goes back to RAG.
    project.chat(messages=[{"role": "user", "content": "login endpoint again?"}])
    assert doc_messages(fake_llm[-1]) == []
