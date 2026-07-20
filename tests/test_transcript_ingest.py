"""Claude Code session capture: transcript parsing, ingestion, idempotency,
and the SessionEnd hook CLI entry point."""
from __future__ import annotations

import io
import json

from tests.fakes import make_engine
from tokensense.server.cli import build_parser, run_ingest_transcript
from tokensense.server.transcript import MAX_TURN_CHARS, extract_turns, summarize_turns


def _line(type_, role, content, sidechain=False):
    return json.dumps(
        {"type": type_, "isSidechain": sidechain, "message": {"role": role, "content": content}}
    )


def write_transcript(tmp_path, lines, name="session.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


REAL_SHAPE_LINES = [
    json.dumps({"type": "queue-operation", "operation": "x"}),
    _line(
        "user",
        "user",
        [
            {"type": "text", "text": "<system-reminder>injected scaffolding</system-reminder>"},
            {"type": "text", "text": "Let's use JWT for the auth flow"},
        ],
    ),
    _line(
        "assistant",
        "assistant",
        [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "text", "text": "Done - JWT with a 15 minute expiry in login.py"},
        ],
    ),
    _line("user", "user", [{"type": "tool_result", "tool_use_id": "t1", "content": "huge output"}]),
    _line("assistant", "assistant", [{"type": "text", "text": "sidechain noise"}], sidechain=True),
    json.dumps({"type": "file-history-snapshot", "snapshot": {}}),
    "{not valid json",
    _line("user", "user", "plain string reply works too"),
]


def test_extract_turns_keeps_conversation_drops_noise(tmp_path):
    turns = extract_turns(write_transcript(tmp_path, REAL_SHAPE_LINES))
    assert turns == [
        {"role": "user", "content": "Let's use JWT for the auth flow"},
        {"role": "assistant", "content": "Done - JWT with a 15 minute expiry in login.py"},
        {"role": "user", "content": "plain string reply works too"},
    ]


def test_extract_turns_truncates_runaway_turns(tmp_path):
    lines = [_line("user", "user", [{"type": "text", "text": "x" * (MAX_TURN_CHARS * 2)}])]
    (turn,) = extract_turns(write_transcript(tmp_path, lines))
    assert len(turn["content"]) == MAX_TURN_CHARS


def test_summarize_turns_chains_batches():
    class RecordingSummarizer:
        def __init__(self):
            self.calls = []

        def summarize(self, prior, turns):
            self.calls.append((prior, len(turns)))
            return f"s{len(self.calls)}"

    summarizer = RecordingSummarizer()
    turns = [{"role": "user", "content": str(i)} for i in range(45)]
    assert summarize_turns(summarizer, turns, batch_size=20) == "s3"
    assert summarizer.calls == [(None, 20), ("s1", 20), ("s2", 5)]


TURNS = [
    {"role": "user", "content": "Let's use JWT for the auth flow"},
    {"role": "assistant", "content": "Done, 15 minute expiry"},
]


def test_ingest_stores_retrievable_memory(monkeypatch):
    engine = make_engine(monkeypatch)
    summary = engine.ingest_transcript_turns(TURNS, project_name="proj-x", external_id="sess-1")
    assert summary and "JWT" in summary

    project = engine.store.get_or_create_project("proj-x")
    assert [c.content for c in engine.store.memory_chunks] == [summary]
    assert engine.store.top_k_memory_chunks(project.id, [0.0]) == [summary]
    assert engine.stats()["sessions"] == 1


def test_reingest_same_session_replaces_instead_of_duplicating(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.ingest_transcript_turns(TURNS, project_name="proj-x", external_id="sess-1")
    grown = TURNS + [{"role": "user", "content": "also add rate limiting"}]
    summary2 = engine.ingest_transcript_turns(grown, project_name="proj-x", external_id="sess-1")

    assert [c.content for c in engine.store.memory_chunks] == [summary2]
    assert len(engine.store.sub_conversations) == 1
    assert engine.stats()["sessions"] == 1  # one session, not two

    engine.ingest_transcript_turns(TURNS, project_name="proj-x", external_id="sess-2")
    assert len(engine.store.memory_chunks) == 2
    assert engine.stats()["sessions"] == 2


def test_thin_transcripts_are_not_stored(monkeypatch):
    engine = make_engine(monkeypatch)
    assert engine.ingest_transcript_turns([], project_name="p") is None
    assert engine.ingest_transcript_turns(TURNS[:1], project_name="p") is None
    assert engine.store.memory_chunks == []


def test_project_resolution_falls_back_to_cwd(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch)
    monkeypatch.delenv("TOKENSENSE_PROJECT", raising=False)
    workdir = tmp_path / "my-repo"
    (workdir / ".git").mkdir(parents=True)
    engine.ingest_transcript_turns(TURNS, cwd=str(workdir), external_id="s")
    assert "my-repo" in engine.store.projects


def test_cli_hook_mode_reads_stdin_payload(monkeypatch, tmp_path):
    transcript = write_transcript(tmp_path, REAL_SHAPE_LINES)
    hook_payload = {"session_id": "cc-sess", "transcript_path": str(transcript), "cwd": str(tmp_path)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(hook_payload)))
    monkeypatch.setenv("TOKENSENSE_DB_URL", "unused")

    engines = []

    def factory(config):
        engine = make_engine(monkeypatch)
        engines.append(engine)
        return engine

    args = build_parser().parse_args(["ingest-transcript"])
    assert run_ingest_transcript(args, engine_factory=factory) == 0

    (engine,) = engines
    (sub,) = engine.store.sub_conversations.values()
    assert sub.external_id == "cc-sess"
    assert len(engine.store.memory_chunks) == 1


def test_cli_never_fails_the_host_hook(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    args = build_parser().parse_args(["ingest-transcript"])
    assert run_ingest_transcript(args) == 0

    monkeypatch.setenv("TOKENSENSE_DB_URL", "unused")
    args = build_parser().parse_args(["ingest-transcript", "/nonexistent/transcript.jsonl"])

    def exploding_factory(config):
        raise RuntimeError("db down")

    assert run_ingest_transcript(args, engine_factory=exploding_factory) == 0
