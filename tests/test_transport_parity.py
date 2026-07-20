"""Phase 6 sanity check: the proxy transport and the direct engine (SDK-shaped)
path produce identical payloads and identical tracker numbers for the same
scenario — the shared core is actually shared, not silently diverging."""
import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tokensense.server.proxy import create_app

from .fakes import make_engine

TURNS = ["we decided on pgvector", "the port is 8317", "what did we decide so far?"]


def _run_direct(monkeypatch):
    """Drive the engine the way the SDK's Project.chat does."""
    engine = make_engine(monkeypatch)
    session = engine.get_session("s", project_header="parity")
    payloads = []
    for turn in TURNS:
        payload, _ = engine.prepare_payload(session, turn)
        payloads.append(payload)
        engine.record_turn(
            session, current_message=turn, assistant_content=f"re: {turn}", payload=payload, model="gpt-4o"
        )
    return engine.stats(), payloads


def _run_proxy(monkeypatch):
    forwarded = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        forwarded.append(body["messages"])
        last_user = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": f"re: {last_user}"}}]},
        )

    engine = make_engine(monkeypatch)
    app = create_app(engine, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with TestClient(app) as client:
        for turn in TURNS:
            client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": turn}]},
                headers={"x-tokensense-session": "s", "x-tokensense-project": "parity"},
            )
    return engine.stats(), forwarded


def test_proxy_and_direct_paths_are_identical(monkeypatch):
    direct_stats, direct_payloads = _run_direct(monkeypatch)
    proxy_stats, proxy_payloads = _run_proxy(monkeypatch)

    assert direct_payloads == proxy_payloads
    direct_stats.pop("sessions"), proxy_stats.pop("sessions")
    assert direct_stats == proxy_stats
