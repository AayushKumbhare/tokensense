"""Live tests of the fully local stack: Ollama summarization (qwen2.5:3b) and
Ollama embeddings (nomic-embed-text) against the Docker Postgres.

Complements test_live_integration.py: same pipeline, but the summaries stored
in memory are genuine model output and the embeddings are real nomic vectors —
no API keys anywhere. Opt-in via TOKENSENSE_LIVE_OLLAMA=1 (real model calls,
seconds-to-minutes), and additionally requires both the Docker database and an
Ollama server with the default models to be reachable.
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid

import pytest
from sqlalchemy import text

from tests.test_live_integration import DB_URL, _db_available
from tokensense.client import TokenSenseClient
from tokensense.memory.embedder import Embedder
from tokensense.summarizers.ollama import OllamaSummarizer

import tokensense.project as project_module

OLLAMA_URL = "http://localhost:11434"
SUMMARIZER_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"


def _ollama_models_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
        return all(
            any(name.startswith(wanted) for name in names)
            for wanted in (SUMMARIZER_MODEL, EMBEDDING_MODEL)
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENSENSE_LIVE_OLLAMA")
    or not (_db_available() and _ollama_models_available()),
    reason=(
        "set TOKENSENSE_LIVE_OLLAMA=1 with live Postgres (docker compose up -d) and "
        f"Ollama serving {SUMMARIZER_MODEL} + {EMBEDDING_MODEL}"
    ),
)


def test_summarizer_extracts_facts_live():
    """The default summarizer, against real Ollama, produces a summary that
    retains at least one of the conversation's concrete facts."""
    summarizer = OllamaSummarizer()
    summary = summarizer.summarize(
        None,
        [
            {"role": "user", "content": "Let's use JWT tokens for the auth flow in login.py"},
            {"role": "assistant", "content": "Agreed - JWT with a 15 minute expiry, refresh via /token/refresh"},
        ],
    )
    assert isinstance(summary, str) and summary.strip()
    facts = ("jwt", "15", "login.py", "refresh")
    assert any(f in summary.lower() for f in facts), f"no conversation fact survived: {summary!r}"


def test_end_to_end_fully_local(monkeypatch):
    """Full smoke run on the fully local stack: real nomic embeddings, real
    qwen summarization, live Postgres — only the chat model is faked. Session
    1's memory must come back in session 2's payload, and semantic ranking
    must put the relevant chunk first."""
    captured_payloads: list[list[dict]] = []

    # tokensense.project and the summarizer import the same litellm module
    # object, so a blanket patch would fake the summarizer too. Route ollama/
    # models (the summarizer) to the real implementation; fake only the chat
    # model and capture its payloads.
    real_completion = project_module.litellm.completion

    def fake_completion(model=None, messages=None, api_key=None, **kwargs):
        if str(model).startswith("ollama/"):
            return real_completion(model=model, messages=messages, **kwargs)
        captured_payloads.append(messages)
        return {"choices": [{"message": {"content": "Sounds good, JWT it is."}}]}

    monkeypatch.setattr(project_module.litellm, "completion", fake_completion)
    monkeypatch.setattr(
        project_module.litellm,
        "token_counter",
        lambda model=None, messages=None: sum(len(str(m.get("content", ""))) for m in messages),
    )

    def make_client() -> TokenSenseClient:
        client = TokenSenseClient(provider="anthropic", api_key="test-key", db_url=DB_URL)
        # Defaults must already be the local stack — no overrides.
        assert isinstance(client.summarizer, OllamaSummarizer)
        assert isinstance(client.embedder, Embedder) and client.embedder.model.startswith("ollama/")
        return client

    project_name = f"live-local-{uuid.uuid4().hex[:8]}"
    client1 = make_client()
    try:
        # Session 1: two topics, ended as two sub-conversations, so session 2
        # has genuinely competing memory chunks to rank.
        project1 = client1.project(project_name)
        project1.chat(messages=[{"role": "user", "content": "Let's use JWT tokens for the auth flow"}])
        project1.end_session()
        project1.chat(messages=[{"role": "user", "content": "The deploy pipeline should build Docker images tagged by git sha"}])
        project1.end_session()

        with client1.store.Session() as session:
            stored = session.execute(
                text(
                    "SELECT mc.content FROM memory_chunks mc"
                    " JOIN projects p ON p.id = mc.project_id WHERE p.name = :name"
                ),
                {"name": project_name},
            ).scalars().all()
        assert len(stored) == 2 and all(s.strip() for s in stored)

        # Session 2, fresh client: ask about auth; real nomic embeddings must
        # rank the auth summary above the deploy summary in the context block.
        client2 = make_client()
        project2 = client2.project(project_name)
        project2.chat(messages=[{"role": "user", "content": "What did we decide about the auth flow and tokens?"}])

        context_messages = [
            m["content"]
            for m in captured_payloads[-1]
            if m["role"] == "system" and "Relevant context from past sessions" in m["content"]
        ]
        assert len(context_messages) == 1
        context = context_messages[0].lower()
        assert "jwt" in context or "auth" in context
        auth_pos = min((context.find(w) for w in ("jwt", "auth") if w in context), default=-1)
        deploy_pos = context.find("deploy") if "deploy" in context else context.find("docker")
        if deploy_pos != -1:
            assert auth_pos < deploy_pos, "auth memory should be ranked above deploy memory"
    finally:
        with client1.store.Session() as session:
            session.execute(text("DELETE FROM projects WHERE name = :name"), {"name": project_name})
            session.commit()
