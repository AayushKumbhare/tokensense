"""MCP transport: project memory exposed as callable tools (see revised
architecture plan, Phase 2).

Retrieval trigger policy (resolved open decision): agent-invoked. MCP tools
are called by the host agent when it judges them relevant, so retrieval fires
only when the agent asks — more token-efficient than the proxy's unconditional
per-turn retrieval, at the cost of depending on the host tool actually calling
the tool. The tool descriptions below are written to make that reliable. Tools
run unconditionally per-turn only on the proxy transport, which sits in the
request path and can.

One MCP server subprocess == one session. The project is resolved once at
startup from the subprocess's environment and working directory (host tools
launch MCP servers in the project directory, so cwd inference works
naturally), and stays bound until exit or an explicit switch_project.
end_session auto-fires on subprocess exit via atexit — no explicit call
required from the host tool.
"""
from __future__ import annotations

import atexit
import os
import uuid

from .engine import ServerEngine

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The MCP transport requires the 'mcp' package. Install with: pip install 'tokensense[server]'"
    ) from exc


def create_mcp_server(engine: ServerEngine, *, session_id: str | None = None) -> FastMCP:
    session_id = session_id or f"mcp-{uuid.uuid4()}"
    mcp = FastMCP("tokensense")

    def _session():
        return engine.get_session(session_id, cwd=os.getcwd())

    @mcp.tool()
    def get_project_context(query: str) -> str:
        """Retrieve relevant context from past sessions in this project. Call
        this whenever the user refers to prior work, earlier decisions, or
        anything not visible in the current conversation — before answering
        from memory or guessing."""
        session = _session()
        retrieved = engine.retrieve_context(session, query)
        if not retrieved:
            return f"No stored context found in project '{session.project.name}' for this query."
        chunks = "\n\n---\n\n".join(retrieved)
        return f"Context from past sessions in project '{session.project.name}':\n\n{chunks}"

    @mcp.tool()
    def save_context(note: str) -> str:
        """Save an important decision, outcome, or constraint to project
        memory immediately, without waiting for session end. Call this when
        the user asks to remember something, or right after a decision worth
        keeping is made mid-session."""
        session = _session()
        engine.save_context(session, note)
        return (
            f"Saved to project '{session.project.name}' memory — retrievable via "
            "get_project_context in this and future sessions."
        )

    @mcp.tool()
    def stats() -> dict:
        """Report TokenSense token and CO2 savings for this server session:
        tokens sent as retrieved summaries vs. the raw session tokens they
        replaced. Call when the user asks about savings, stats, or impact."""
        return engine.stats()

    @mcp.tool()
    def end_session(summary: str = "") -> str:
        """End the current TokenSense session, persisting its memory for future
        sessions. Optionally pass a summary of decisions and outcomes from this
        conversation — since the host tool owns the conversation, this is how
        its content reaches project memory."""
        session = _session()
        if summary:
            session.raw_turns.append({"role": "user", "content": f"Session notes: {summary}"})
            session.window.add_turn({"role": "user", "content": f"Session notes: {summary}"})
        stored = engine.end_session(session_id)
        if stored is None:
            return "Session ended; nothing to store (no turns or notes were recorded)."
        return f"Session ended; memory stored for project '{session.project.name}'."

    @mcp.tool()
    def switch_project(project: str) -> str:
        """Deliberately switch this session to a different TokenSense project.
        Ends (and persists) the current session first — a session is never
        silently retargeted."""
        session = engine.switch_project(session_id, project)
        return f"Switched to project '{session.project.name}'. Previous session was ended and persisted."

    atexit.register(engine.end_all)
    return mcp
