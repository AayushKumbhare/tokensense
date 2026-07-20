"""Benchmark harness: run the scenarios through both entry points — the direct
engine path (SDK-shaped) and the HTTP proxy — and print a transport-comparison
table (see revised architecture plan, Phase 6).

The upstream *provider* call is stubbed in every mode (assistant replies are
deterministic), because the quantities under test — retrieval recall, payload
compression, and tracker parity across transports — don't depend on what the
provider says back. What differs by mode is everything else:

  --offline   in-memory store + fake embedder/summarizer; no DB, no network.
              Validates plumbing and transport parity; compression numbers are
              not meaningful (the fake summarizer echoes its input).
  --db-url    real Postgres + real embeddings + real summarizer (Ollama by
              default). Recall and compression numbers are real.

Usage:
    python benchmarks/run.py --offline
    python benchmarks/run.py --db-url postgresql://localhost/tokensense_bench
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.scenarios import SCENARIOS

MODEL = "gpt-4o"


def make_offline_engine():
    from tests.fakes import FakeEmbedder, FakeStore, FakeSummarizer
    from tokensense.memory.retriever import Retriever
    from tokensense.server.config import ServerConfig
    from tokensense.server.engine import ServerEngine

    engine = ServerEngine(ServerConfig(db_url="offline"), store=FakeStore())
    engine.embedder = FakeEmbedder()
    engine.retriever = Retriever(engine.store, engine.embedder, top_k=5)
    engine.summarizer = FakeSummarizer()
    return engine


def make_live_engine(db_url: str):
    from tokensense.server.config import ServerConfig
    from tokensense.server.engine import ServerEngine

    return ServerEngine(ServerConfig(db_url=db_url))


def probe_recall(retrieved: list[str], expect: list[str]) -> float:
    hits = sum(1 for needle in expect if any(needle in chunk for chunk in retrieved))
    return hits / len(expect)


def run_direct(engine, scenario, project: str) -> dict:
    for i, turns in enumerate(scenario["sessions"]):
        session_id = f"{project}-s{i}"
        session = engine.get_session(session_id, project_header=project)
        for turn in turns:
            payload, _ = engine.prepare_payload(session, turn)
            engine.record_turn(
                session,
                current_message=turn,
                assistant_content=f"Acknowledged: {turn}",
                payload=payload,
                model=MODEL,
            )
        engine.end_session(session_id)

    probe_session = engine.get_session(f"{project}-probe", project_header=project)
    _, retrieved = engine.prepare_payload(probe_session, scenario["probe"])
    return {"recall": probe_recall(retrieved, scenario["expect"]), **engine.stats()}


def run_proxy(engine, scenario, project: str) -> dict:
    import httpx
    from fastapi.testclient import TestClient

    from tokensense.server.proxy import create_app

    retrieved_by_probe: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        last_user = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": f"Acknowledged: {last_user}"}}]},
        )

    app = create_app(engine, http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    with TestClient(app) as client:
        def post(text: str, session_id: str):
            return client.post(
                "/v1/chat/completions",
                json={"model": MODEL, "messages": [{"role": "user", "content": text}]},
                headers={"x-tokensense-session": session_id, "x-tokensense-project": project},
            )

        for i, turns in enumerate(scenario["sessions"]):
            session_id = f"{project}-s{i}"
            for turn in turns:
                post(turn, session_id)
            engine.end_session(session_id)

        probe_session = engine.get_session(f"{project}-probe", project_header=project)
        _, retrieved_by_probe = engine.prepare_payload(probe_session, scenario["probe"])

    return {"recall": probe_recall(retrieved_by_probe, scenario["expect"]), **engine.stats()}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="fakes only: no DB, no network")
    mode.add_argument("--db-url", help="Postgres URL for a live run")
    args = parser.parse_args()

    stamp = int(time.time())
    rows = []
    for scenario in SCENARIOS:
        for transport, runner in [("direct", run_direct), ("proxy", run_proxy)]:
            engine = make_offline_engine() if args.offline else make_live_engine(args.db_url)
            project = f"bench-{scenario['name']}-{transport}-{stamp}"
            result = runner(engine, scenario, project)
            rows.append({"scenario": scenario["name"], "transport": transport, **result})

    header = f"{'scenario':<34} {'transport':<9} {'sent':>8} {'baseline':>9} {'saved':>8} {'saved%':>7} {'recall':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        pct = 100 * row["tokens_saved"] / row["tokens_baseline"] if row["tokens_baseline"] else 0.0
        print(
            f"{row['scenario']:<34} {row['transport']:<9} {row['tokens_sent']:>8} "
            f"{row['tokens_baseline']:>9} {row['tokens_saved']:>8} {pct:>6.1f}% {row['recall']:>7.2f}"
        )

    print()
    for scenario in SCENARIOS:
        pair = [r for r in rows if r["scenario"] == scenario["name"]]
        direct, proxy = pair[0], pair[1]
        match = direct["tokens_sent"] == proxy["tokens_sent"] and direct["tokens_baseline"] == proxy["tokens_baseline"]
        status = "MATCH" if match else "DIVERGED"
        print(f"tracker parity [{scenario['name']}]: {status}")


if __name__ == "__main__":
    main()
