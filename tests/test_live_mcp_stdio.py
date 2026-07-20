"""True end-to-end MCP test: a real `tokensense serve --mcp` subprocess,
spoken to over the actual MCP stdio protocol — the same path Claude Code uses.

Seeds project memory through the transcript-ingestion pipeline (real qwen
summarization, real nomic embeddings, live Postgres), then verifies a fresh
MCP server process retrieves it via get_project_context. Gated like the other
fully-local live tests: TOKENSENSE_LIVE_OLLAMA=1 plus running services.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.test_live_integration import DB_URL, _db_available
from tests.test_live_ollama import _ollama_models_available
from tokensense.server.config import ServerConfig
from tokensense.server.engine import ServerEngine

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENSENSE_LIVE_OLLAMA")
    or not (_db_available() and _ollama_models_available()),
    reason="set TOKENSENSE_LIVE_OLLAMA=1 with live Postgres and Ollama serving the default models",
)

TOKENSENSE_BIN = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "tokensense"

SESSION_TURNS = [
    {"role": "user", "content": "Let's use JWT tokens for the auth flow"},
    {"role": "assistant", "content": "Done - JWT with a 15 minute expiry in login.py"},
]


async def _mcp_roundtrip(project_name: str) -> tuple[list[str], str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import get_default_environment, stdio_client

    params = StdioServerParameters(
        command=str(TOKENSENSE_BIN),
        args=["serve", "--mcp"],
        env={
            **get_default_environment(),
            "TOKENSENSE_DB_URL": DB_URL,
            "TOKENSENSE_PROJECT": project_name,
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            result = await session.call_tool(
                "get_project_context", {"query": "What did we decide about the auth flow?"}
            )
            return tools, result.content[0].text


def test_claude_code_shaped_roundtrip():
    """Prior session captured by the hook pipeline -> fresh MCP server process
    -> get_project_context returns the decision. This is the Phase 2 exit
    criterion run against real processes instead of mocks."""
    project_name = f"live-mcp-{uuid.uuid4().hex[:8]}"
    engine = ServerEngine(ServerConfig(db_url=DB_URL))

    try:
        # Session 1: what the SessionEnd hook does with a finished transcript.
        summary = engine.ingest_transcript_turns(
            SESSION_TURNS, project_name=project_name, external_id="cc-live-e2e"
        )
        assert summary is not None

        # Session 2: a brand-new MCP server subprocess, as Claude Code launches it.
        tools, context = asyncio.run(asyncio.wait_for(_mcp_roundtrip(project_name), timeout=90))

        assert {"get_project_context", "end_session", "switch_project"} <= set(tools)
        assert project_name in context
        assert "jwt" in context.lower()
    finally:
        with engine.store.Session() as session:
            session.execute(text("DELETE FROM projects WHERE name = :name"), {"name": project_name})
            session.commit()
