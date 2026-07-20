"""Phase 1: proxy transport — payload rewriting, passthrough, key forwarding."""
import json

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tokensense.server.proxy import create_app

from .fakes import make_engine

OPENAI_RESPONSE = {
    "id": "chatcmpl-1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello from upstream"}}],
    "usage": {"total_tokens": 10},
}

ANTHROPIC_RESPONSE = {
    "id": "msg-1",
    "content": [{"type": "text", "text": "hello from upstream"}],
    "usage": {"input_tokens": 5, "output_tokens": 5},
}


@pytest.fixture
def upstream():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = OPENAI_RESPONSE if "chat/completions" in str(request.url) else ANTHROPIC_RESPONSE
        return httpx.Response(200, json=body)

    return captured, httpx.MockTransport(handler)


@pytest.fixture
def client(monkeypatch, upstream):
    _, transport = upstream
    engine = make_engine(monkeypatch)
    app = create_app(engine, http_client=httpx.AsyncClient(transport=transport))
    with TestClient(app) as test_client:
        test_client.engine = engine
        yield test_client


def _openai_request(client, text, session="s1", project="alpha"):
    return client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": text}]},
        headers={
            "authorization": "Bearer sk-test",
            "x-tokensense-session": session,
            "x-tokensense-project": project,
        },
    )


def test_openai_passthrough_and_headers(client, upstream):
    response = _openai_request(client, "hi")
    assert response.status_code == 200
    assert response.json() == OPENAI_RESPONSE  # body untouched (passthrough decision)
    assert response.headers["x-tokensense-project"] == "alpha"
    assert "x-tokensense-retrieved" in response.headers
    assert "x-tokensense-tokens-saved" in response.headers


def test_api_key_forwarded_not_persisted(client, upstream):
    captured, _ = upstream
    _openai_request(client, "hi")
    assert captured[0].headers["authorization"] == "Bearer sk-test"
    # Nothing engine-side holds the key.
    engine = client.engine
    assert "sk-test" not in json.dumps(engine.store.__dict__, default=str)


def test_memory_augmented_payload_reaches_upstream(client, upstream):
    captured, _ = upstream
    # Store a memory chunk in alpha, then start a new session and ask.
    _openai_request(client, "we chose port 8317 for the proxy", session="s1")
    client.engine.end_session("s1")

    _openai_request(client, "which port did we choose?", session="s2")
    body = json.loads(captured[-1].content)
    system_messages = [m for m in body["messages"] if m["role"] == "system"]
    assert system_messages, "expected retrieved context in the forwarded payload"
    assert any("8317" in m["content"] for m in system_messages)
    assert body["messages"][-1] == {"role": "user", "content": "which port did we choose?"}


def test_anthropic_system_moved_to_top_level(client, upstream):
    captured, _ = upstream
    # Seed memory so the payload includes a system context block.
    _openai_request(client, "decision: use redis for queues", session="s1")
    client.engine.end_session("s1")

    response = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "max_tokens": 100,
              "messages": [{"role": "user", "content": "what did we decide?"}]},
        headers={"x-api-key": "sk-ant-test", "x-tokensense-session": "s3", "x-tokensense-project": "alpha"},
    )
    assert response.status_code == 200
    assert response.json() == ANTHROPIC_RESPONSE
    body = json.loads(captured[-1].content)
    assert all(m["role"] != "system" for m in body["messages"])
    assert "redis" in body["system"]
    assert captured[-1].headers["x-api-key"] == "sk-ant-test"


def test_streaming_rejected(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert "stream" in response.json()["error"]["message"]


def test_sliding_window_carries_within_session(client, upstream):
    captured, _ = upstream
    _openai_request(client, "first turn", session="s9")
    _openai_request(client, "second turn", session="s9")
    body = json.loads(captured[-1].content)
    contents = [m["content"] for m in body["messages"]]
    assert "first turn" in contents  # prior verbatim turn included
    assert contents[-1] == "second turn"


def test_upstream_error_passes_through_without_recording(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    engine = make_engine(monkeypatch)
    app = create_app(engine, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-tokensense-session": "s1", "x-tokensense-project": "alpha"},
        )
        assert response.status_code == 401
        session = engine.get_session("s1", project_header="alpha")
        assert not session.has_turns  # failed calls don't pollute the window


def test_stats_endpoint(client):
    _openai_request(client, "hi")
    stats = client.get("/stats").json()
    assert set(stats) >= {"tokens_sent", "tokens_baseline", "tokens_saved", "co2_saved_grams"}
