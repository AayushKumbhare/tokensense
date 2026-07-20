"""Phase 2: MCP transport — tools exercised through the FastMCP tool registry."""
import asyncio

import pytest

pytest.importorskip("mcp")

from tokensense.server.mcp_server import create_mcp_server

from .fakes import make_engine


def _call(server, tool, arguments):
    result = asyncio.run(server.call_tool(tool, arguments))
    # FastMCP returns a list of content blocks (possibly with structured output).
    blocks = result[0] if isinstance(result, tuple) else result
    return "\n".join(b.text for b in blocks if hasattr(b, "text"))


@pytest.fixture
def setup(monkeypatch):
    engine = make_engine(monkeypatch)
    server = create_mcp_server(engine, session_id="mcp-test")
    return engine, server


def test_tools_registered(setup):
    _, server = setup
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert tools == {"get_project_context", "save_context", "end_session", "switch_project", "stats"}


def test_get_project_context_returns_prior_session_memory(setup, monkeypatch, tmp_path):
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    # A prior session left a memory chunk in alpha.
    prior = engine.get_session("prior", project_header="alpha")
    payload, _ = engine.prepare_payload(prior, "we picked HNSW indexes")
    engine.record_turn(
        prior, current_message="we picked HNSW indexes", assistant_content="noted", payload=payload, model="gpt-4o"
    )
    engine.end_session("prior")

    text = _call(server, "get_project_context", {"query": "which index type?"})
    assert "HNSW" in text
    assert "alpha" in text


def test_save_context_persists_immediately(setup, monkeypatch):
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    text = _call(server, "save_context", {"note": "we chose HNSW over IVFFlat"})
    assert "alpha" in text
    assert len(engine.store.memory_chunks) == 1
    assert "HNSW" in engine.store.memory_chunks[0].content

    # Retrievable in the same session, before any session end.
    ctx = _call(server, "get_project_context", {"query": "which index type?"})
    assert "HNSW" in ctx

    # Session end doesn't re-store the note (it's persisted, not window-folded).
    _call(server, "end_session", {})
    assert len(engine.store.memory_chunks) == 1


def test_get_project_context_logs_savings(setup, monkeypatch):
    """MCP savings model: retrieved-summary tokens vs. the raw session tokens
    they replace (token counting is patched to char counts in make_engine)."""
    import json

    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    project = engine.store.get_or_create_project("alpha")
    sub = engine.store.start_sub_conversation(project.id, external_id="cc-1")
    raw = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 400}]
    engine.store.update_sub_conversation(sub.id, raw_turns=json.dumps(raw))
    summary = "short summary of the long session"
    engine.store.add_memory_chunk(project.id, sub.id, summary, [1.0, 0.0, 0.0])

    _call(server, "get_project_context", {"query": "what happened last time?"})
    stats = engine.stats()
    assert stats["tokens_baseline"] == 800
    assert stats["tokens_sent"] == len(summary)
    assert stats["tokens_saved"] == 800 - len(summary)


def test_savings_skip_sessions_without_raw_turns(setup, monkeypatch):
    """A note-only chunk replaced no raw session, so it must log nothing."""
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    _call(server, "save_context", {"note": "a bare note"})
    _call(server, "get_project_context", {"query": "anything"})
    stats = engine.stats()
    assert stats["tokens_baseline"] == 0
    assert stats["tokens_sent"] == 0


def test_stats_tool_reports_tracker(setup, monkeypatch):
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    engine.tracker.log_call(actual_tokens=100, baseline_tokens=350)
    text = _call(server, "stats", {})
    assert "tokens_saved" in text
    assert "250" in text
    assert "co2_saved_grams" in text


def test_end_session_with_summary_persists_memory(setup, monkeypatch):
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    text = _call(server, "end_session", {"summary": "decided to defer partitioning"})
    assert "memory stored" in text
    assert len(engine.store.memory_chunks) == 1
    assert "defer partitioning" in engine.store.memory_chunks[0].content


def test_end_session_empty_stores_nothing(setup, monkeypatch):
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    text = _call(server, "end_session", {})
    assert "nothing to store" in text
    assert engine.store.memory_chunks == []


def test_switch_project_is_explicit_and_persists_old(setup, monkeypatch):
    engine, server = setup
    monkeypatch.setenv("TOKENSENSE_PROJECT", "alpha")
    session = engine.get_session("mcp-test")
    alpha_id = session.project.id
    session.raw_turns.append({"role": "user", "content": "alpha work"})
    session.window.add_turn({"role": "user", "content": "alpha work"})

    text = _call(server, "switch_project", {"project": "beta"})
    assert "beta" in text
    assert engine.bindings.get("mcp-test") == "proj-beta"
    assert [c.project_id for c in engine.store.memory_chunks] == [alpha_id]

    # Env var change alone must never retarget the session (Phase 4 guard).
    monkeypatch.setenv("TOKENSENSE_PROJECT", "gamma")
    assert engine.get_session("mcp-test").project.name == "beta"
