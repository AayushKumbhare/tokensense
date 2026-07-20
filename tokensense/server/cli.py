"""`tokensense serve` — launch the proxy (default) or MCP transport.
`tokensense ingest-transcript` — Claude Code SessionEnd / PreCompact hook entry
point (both hooks send the same {session_id, transcript_path, cwd} payload, and
ingestion is idempotent per session, so one command serves both)."""
from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_PORT, ServerConfig
from .engine import ServerEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokensense")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the TokenSense server")
    serve.add_argument("--mcp", action="store_true", help="Run as an MCP server on stdio instead of the HTTP proxy")
    serve.add_argument("--port", type=int, default=None, help=f"Proxy port (default {DEFAULT_PORT})")
    serve.add_argument("--db-url", default=None, help="Postgres URL (default: $TOKENSENSE_DB_URL)")
    serve.add_argument("--project", default=None, help="Explicit project name (overrides env/cwd inference)")

    ingest = subparsers.add_parser(
        "ingest-transcript",
        help="Summarize a Claude Code session transcript into project memory "
        "(with no path, reads the SessionEnd/PreCompact hook JSON from stdin)",
    )
    ingest.add_argument("path", nargs="?", default=None, help="Transcript JSONL path (omit when run as a hook)")
    ingest.add_argument("--db-url", default=None, help="Postgres URL (default: $TOKENSENSE_DB_URL)")
    ingest.add_argument("--project", default=None, help="Explicit project name (overrides env/cwd inference)")
    return parser


def _load_config(args) -> ServerConfig:
    try:
        config = ServerConfig.from_env()
    except ValueError:
        if not args.db_url:
            print("error: set TOKENSENSE_DB_URL or pass --db-url", file=sys.stderr)
            raise SystemExit(2)
        config = ServerConfig(db_url=args.db_url)
    if args.db_url:
        config.db_url = args.db_url
    return config


def run_ingest_transcript(args, engine_factory=ServerEngine) -> int:
    """Ingest one transcript. Hook mode (no path) reads Claude Code's hook JSON
    — {session_id, transcript_path, cwd, ...}, same shape for SessionEnd and
    PreCompact — from stdin. Always exits 0: a capture failure must never break
    the host tool."""
    path, cwd, external_id = args.path, None, None
    if path is None:
        try:
            hook_payload = json.load(sys.stdin)
            path = hook_payload["transcript_path"]
            cwd = hook_payload.get("cwd")
            external_id = hook_payload.get("session_id")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"tokensense: no transcript path (bad hook payload: {exc})", file=sys.stderr)
            return 0
    if external_id is None:
        # Claude Code names transcripts <session-id>.jsonl, so path-mode
        # re-ingestion of the same session stays idempotent too.
        from pathlib import Path

        external_id = Path(path).stem or None

    try:
        from .transcript import extract_turns

        engine = engine_factory(_load_config(args))
        turns = extract_turns(path)
        summary = engine.ingest_transcript_turns(
            turns, project_name=args.project, cwd=cwd, external_id=external_id
        )
    except Exception as exc:
        print(f"tokensense: transcript ingestion failed: {exc}", file=sys.stderr)
        return 0
    if summary is None:
        print("tokensense: session too thin to store")
    else:
        print(f"tokensense: session memory stored ({len(turns)} turns)")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "ingest-transcript":
        raise SystemExit(run_ingest_transcript(args))

    config = _load_config(args)
    if args.port:
        config.port = args.port

    engine = ServerEngine(config)

    if args.mcp:
        from .mcp_server import create_mcp_server

        # --project pins the whole subprocess to one project, skipping cwd inference.
        session_id = None
        if args.project:
            session = engine.switch_project("mcp-pinned", args.project)
            session_id = session.session_id
        create_mcp_server(engine, session_id=session_id).run()
        return

    import uvicorn

    from .proxy import create_app

    uvicorn.run(create_app(engine), host="127.0.0.1", port=config.port)


if __name__ == "__main__":
    main()
