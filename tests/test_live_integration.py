"""End-to-end smoke test against a live Postgres+pgvector instance (docker-compose.yml).

Runs the real SDK path — Store, Retriever, SlidingWindow, Project — with
deterministic fakes only at the network boundary (chat model, embeddings,
summarizer), so it needs no API keys or Ollama. Skipped automatically when the
database is unreachable; start it with `docker compose up -d --wait`.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import uuid

import pytest
from sqlalchemy import create_engine, text

from tests.fakes import FakeSummarizer
from tokensense.client import TokenSenseClient
from tokensense.memory.retriever import Retriever
from tokensense.memory.store import EMBEDDING_DIM

import tokensense.project as project_module

DB_URL = os.environ.get(
    "TOKENSENSE_TEST_DB_URL", "postgresql://tokensense:tokensense@localhost:5432/tokensense"
)


def _db_available() -> bool:
    try:
        engine = create_engine(DB_URL, connect_args={"connect_timeout": 2})
        with engine.connect():
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="live Postgres not reachable (docker compose up -d)")


class DeterministicEmbedder:
    """Bag-of-hashed-words embeddings: same word -> same vector component, so
    texts sharing words genuinely score closer under cosine distance. Gives the
    HNSW index real similarity structure to rank without any network call."""

    model = "deterministic-test-embedding"

    def embed(self, text_value: str) -> list[float]:
        total = [0.0] * EMBEDDING_DIM
        for word in re.findall(r"\w+", text_value.lower()):
            seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
            rng = random.Random(seed)
            for i in range(EMBEDDING_DIM):
                total[i] += rng.uniform(-1, 1)
        norm = sum(v * v for v in total) ** 0.5 or 1.0
        return [v / norm for v in total]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def live_client_factory(monkeypatch):
    """Builds TokenSenseClients wired to the live DB with faked network edges.
    Captures every payload sent to the (fake) chat model, and cleans up all
    projects created through the factory afterward."""
    captured_payloads: list[list[dict]] = []

    def fake_completion(model=None, messages=None, api_key=None, **kwargs):
        captured_payloads.append(messages)
        return {"choices": [{"message": {"content": f"assistant reply #{len(captured_payloads)}"}}]}

    monkeypatch.setattr(project_module.litellm, "completion", fake_completion)
    monkeypatch.setattr(
        project_module.litellm,
        "token_counter",
        lambda model=None, messages=None: sum(len(str(m.get("content", ""))) for m in messages),
    )

    clients: list[TokenSenseClient] = []

    def make_client() -> TokenSenseClient:
        client = TokenSenseClient(provider="anthropic", api_key="test-key", db_url=DB_URL)
        client.embedder = DeterministicEmbedder()
        client.retriever = Retriever(client.store, client.embedder, top_k=5)
        client.summarizer = FakeSummarizer()
        clients.append(client)
        return client

    yield make_client, captured_payloads

    project_names = {name for client in clients for name in client._projects}
    if clients and project_names:
        with clients[0].store.Session() as session:
            session.execute(
                text("DELETE FROM projects WHERE name = ANY(:names)"),
                {"names": list(project_names)},
            )
            session.commit()


def test_memory_survives_across_sessions(live_client_factory):
    """create project -> chat -> end_session -> a second session (fresh client,
    as if a new process) retrieves the first session's memory from the live DB."""
    make_client, captured_payloads = live_client_factory
    project_name = f"live-smoke-{uuid.uuid4().hex[:8]}"

    # Session 1: establish a decision, then end the session.
    client1 = make_client()
    project1 = client1.project(project_name)
    project1.chat(messages=[{"role": "user", "content": "Let's use JWT tokens for the auth flow"}])
    project1.chat(messages=[{"role": "user", "content": "Agreed, JWT with a 15 minute expiry"}])
    project1.end_session()

    # The summary chunk landed in Postgres with a real 1536-dim embedding.
    with client1.store.Session() as session:
        stored = session.execute(
            text(
                "SELECT mc.content FROM memory_chunks mc"
                " JOIN projects p ON p.id = mc.project_id WHERE p.name = :name"
            ),
            {"name": project_name},
        ).scalars().all()
    assert len(stored) == 1
    assert "JWT" in stored[0]

    # Session 2: a brand-new client (fresh connection pool, no shared state).
    client2 = make_client()
    project2 = client2.project(project_name)
    project2.chat(messages=[{"role": "user", "content": "What did we decide about the auth flow?"}])

    # The last payload sent to the chat model must carry the retrieved memory.
    final_payload = captured_payloads[-1]
    context_messages = [
        m["content"]
        for m in final_payload
        if m["role"] == "system" and "Relevant context from past sessions" in m["content"]
    ]
    assert len(context_messages) == 1
    assert "JWT" in context_messages[0]
    project2.end_session()


def test_document_cache_lifecycle_on_live_db(live_client_factory, tmp_path):
    """Cache decision wiring against real Postgres: a session-added document
    rides verbatim with a persisted TTL (cache write), and after expiry the
    document falls back to RAG chunk retrieval with last_used_at stamped."""
    make_client, captured_payloads = live_client_factory
    project_name = f"live-smoke-doc-{uuid.uuid4().hex[:8]}"

    spec = tmp_path / "api_spec.md"
    spec.write_text("POST /login returns a JWT pair and sets a refresh cookie")

    client1 = make_client()  # provider="anthropic" — supports prompt caching
    project1 = client1.project(project_name)
    doc_id = project1.add_document(str(spec))
    project1.chat(messages=[{"role": "user", "content": "how does login work?"}])

    # Verbatim document with cache markers, TTL persisted to Postgres.
    doc_msgs = [
        m for m in captured_payloads[-1] if m["role"] == "system" and "Project document" in str(m["content"])
    ]
    assert len(doc_msgs) == 1
    assert doc_msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    with client1.store.Session() as session:
        ttl = session.execute(
            text("SELECT provider_ttl_expires_at FROM documents WHERE id = :id"), {"id": doc_id}
        ).scalar()
    assert ttl is not None

    # Expire the TTL in the DB; a fresh client must fall back to RAG chunks.
    with client1.store.Session() as session:
        session.execute(
            text("UPDATE documents SET provider_ttl_expires_at = now() - interval '1 minute' WHERE id = :id"),
            {"id": doc_id},
        )
        session.commit()

    client2 = make_client()
    project2 = client2.project(project_name)
    project2.chat(messages=[{"role": "user", "content": "what does POST /login return?"}])

    payload = captured_payloads[-1]
    assert not any(
        m["role"] == "system" and "Project document" in str(m["content"]) for m in payload
    ), "expired-TTL document must not ride verbatim"
    contexts = [
        m["content"]
        for m in payload
        if m["role"] == "system" and "Relevant context from past sessions" in str(m["content"])
    ]
    assert len(contexts) == 1 and "POST /login" in contexts[0]
    with client2.store.Session() as session:
        last_used = session.execute(
            text("SELECT last_used_at FROM documents WHERE id = :id"), {"id": doc_id}
        ).scalar()
    assert last_used is not None


def test_projects_stay_isolated_on_live_db(live_client_factory):
    """A project retrieves nothing from another project's memory, even when the
    other project's chunk is a much better semantic match for the query."""
    make_client, captured_payloads = live_client_factory
    suffix = uuid.uuid4().hex[:8]

    client = make_client()
    other = client.project(f"live-smoke-other-{suffix}")
    other.chat(messages=[{"role": "user", "content": "The database migration uses alembic revision 42"}])
    other.end_session()

    fresh = client.project(f"live-smoke-fresh-{suffix}")
    fresh.chat(messages=[{"role": "user", "content": "What alembic revision does the database migration use?"}])

    final_payload = captured_payloads[-1]
    assert not any(
        "Relevant context from past sessions" in m["content"] for m in final_payload if m["role"] == "system"
    ), "fresh project must not see the other project's memory"
