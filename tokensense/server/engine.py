"""The shared core behind both server transports (proxy + MCP).

Owns the store, retriever, tracker, summarizer, session bindings, and live
sessions. The proxy and MCP server are thin adapters over this class, so
token/CO2 accounting and memory behavior are identical regardless of
transport — and identical to the SDK path, which uses the same building
blocks (Store, Retriever, SlidingWindow, build_payload, Tracker).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import litellm

from ..client import build_summarizer
from ..memory.embedder import Embedder
from ..memory.retriever import Retriever
from ..memory.store import Store
from ..middleware import SlidingWindow, build_payload
from ..tracker import Tracker
from .config import ServerConfig
from .project_resolve import SessionBindings, resolve_project_name
from .sessions import ServerSession


class ServerEngine:
    def __init__(self, config: ServerConfig, *, store: Store | None = None):
        self.config = config
        self.store = store if store is not None else Store(config.db_url)
        self.embedder = Embedder(model=config.embedding_model)
        self.retriever = Retriever(self.store, self.embedder, top_k=config.top_k)
        self.tracker = Tracker()
        self.summarizer = build_summarizer(config.summarization_model)
        self.bindings = SessionBindings()
        self._sessions: dict[str, ServerSession] = {}
        self._sessions_lock = threading.Lock()

    # -- session lifecycle ---------------------------------------------------------------

    def get_session(
        self,
        session_id: str,
        *,
        project_header: str | None = None,
        cwd: str | None = None,
    ) -> ServerSession:
        """Return the live session, creating and binding it on first use.

        The project is resolved exactly once, at session start; later turns
        reuse the cached binding even if env vars or headers change (Phase 4
        guard — switching mid-session requires an explicit switch_project).
        """
        self.sweep_idle()
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
                return session

            name = resolve_project_name(
                header=project_header, cwd=cwd, default=self.config.default_project
            )
            session = self._start_session(session_id, name)
            self._sessions[session_id] = session
            return session

    def _start_session(self, session_id: str, project_name: str) -> ServerSession:
        project = self.store.get_or_create_project(project_name)
        self.bindings.bind(session_id, project.id)
        return ServerSession(
            session_id=session_id,
            project=project,
            sub_conversation=self.store.start_sub_conversation(project.id),
            window=SlidingWindow(self.summarizer, window_size=self.config.window_size),
        )

    def end_session(self, session_id: str) -> str | None:
        """Summarize and persist the session's memory chunk, then release the
        binding. Returns the summary, or None if the session had no turns."""
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
        self.bindings.release(session_id)
        if session is None:
            return None
        return self._finalize(session)

    def _finalize(self, session: ServerSession) -> str | None:
        if not session.has_turns:
            return None
        summary = session.window.finalize()
        embedding = self.embedder.embed(summary)
        self.store.add_memory_chunk(session.project.id, session.sub_conversation.id, summary, embedding)
        self.store.update_sub_conversation(
            session.sub_conversation.id,
            summary=summary,
            ended_at=datetime.now(timezone.utc),
        )
        self.tracker.log_session_end()
        return summary

    def switch_project(self, session_id: str, project_name: str) -> ServerSession:
        """Deliberate mid-session project switch: ends the current session
        (summarizing it into its original project) before binding to the new one."""
        self.end_session(session_id)
        with self._sessions_lock:
            session = self._start_session(session_id, project_name)
            self._sessions[session_id] = session
            return session

    def sweep_idle(self) -> list[str]:
        """End every session idle past the configured timeout (the proxy has no
        subprocess-exit signal, so idleness stands in for session end)."""
        timeout = self.config.idle_timeout_seconds
        with self._sessions_lock:
            expired = [s.session_id for s in self._sessions.values() if s.idle_for() > timeout]
        for session_id in expired:
            self.end_session(session_id)
        return expired

    def end_all(self) -> None:
        """Shutdown hook: MCP subprocess exit / proxy termination."""
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.end_session(session_id)

    # -- per-turn memory operations --------------------------------------------------------

    def prepare_payload(self, session: ServerSession, current_message: str) -> tuple[list[dict], list[str]]:
        """Retrieve project memory and build the compressed payload for this turn."""
        retrieved = self.retriever.retrieve(session.project.id, current_message)
        payload = build_payload(
            retrieved,
            session.window.rolling_summary,
            session.window.verbatim_turns,
            current_message,
        )
        return payload, retrieved

    def record_turn(
        self,
        session: ServerSession,
        *,
        current_message: str,
        assistant_content: str,
        payload: list[dict],
        model: str,
    ) -> None:
        """Log token savings and fold the turn into the session, exactly as the
        SDK's Project.chat does."""
        baseline_tokens = litellm.token_counter(
            model=model,
            messages=session.raw_turns + [{"role": "user", "content": current_message}],
        )
        actual_tokens = litellm.token_counter(model=model, messages=payload)
        self.tracker.log_call(actual_tokens, baseline_tokens)

        session.raw_turns.append({"role": "user", "content": current_message})
        session.window.add_turn({"role": "user", "content": current_message})
        session.raw_turns.append({"role": "assistant", "content": assistant_content})
        session.window.add_turn({"role": "assistant", "content": assistant_content})

        self.store.update_sub_conversation(
            session.sub_conversation.id, raw_turns=json.dumps(session.raw_turns)
        )
        session.touch()

    # -- MCP-transport operations ----------------------------------------------------------

    def retrieve_context(self, session: ServerSession, query: str) -> list[str]:
        """Agent-invoked retrieval with savings accounting.

        The proxy path counts compressed-payload vs. full-history tokens per
        turn; on MCP the host tool owns the conversation, so the measurable
        counterfactual is instead the raw session tokens each retrieved
        summary stands in for (decisions.md #7). Sessions with no stored raw
        turns (e.g. note-only) replaced nothing and log nothing; document
        chunks likewise pass through unlogged.
        """
        chunks, memory_sources = self.retriever.retrieve_with_sources(session.project.id, query)
        for content, raw_turns_json in memory_sources:
            try:
                raw_turns = json.loads(raw_turns_json or "[]")
            except json.JSONDecodeError:
                continue
            if not raw_turns:
                continue
            # No host model to count against — litellm's default tokenizer is
            # a consistent approximation on both sides of the subtraction.
            baseline = litellm.token_counter(model="", messages=raw_turns)
            actual = litellm.token_counter(model="", messages=[{"role": "user", "content": content}])
            self.tracker.log_call(actual, baseline)
        return chunks

    def save_context(self, session: ServerSession, note: str) -> None:
        """Mid-session persistence: embed and store the note as a retrievable
        memory chunk now, so it survives even if the session never ends
        cleanly. Deliberately not folded into the sliding window — the chunk
        is already stored, and folding it in would duplicate it via the
        end-of-session summary."""
        embedding = self.embedder.embed(note)
        self.store.add_memory_chunk(session.project.id, session.sub_conversation.id, note, embedding)
        session.touch()

    # -- transcript ingestion (Claude Code SessionEnd / PreCompact hooks) ------------------

    MIN_INGEST_TURNS = 2

    def ingest_transcript_turns(
        self,
        turns: list[dict],
        *,
        project_name: str | None = None,
        cwd: str | None = None,
        external_id: str | None = None,
    ) -> str | None:
        """Summarize a host-tool session transcript into project memory.

        This is the deterministic capture path: the host tool (Claude Code)
        owns the conversation, so instead of relying on the agent calling
        end_session with notes, a SessionEnd hook feeds us the transcript.
        Re-ingesting the same `external_id` replaces that session's memory
        instead of duplicating it. Returns the stored summary, or None if the
        transcript was too thin to be worth remembering.
        """
        if len(turns) < self.MIN_INGEST_TURNS:
            return None

        from .transcript import summarize_turns

        name = resolve_project_name(header=project_name, cwd=cwd, default=self.config.default_project)
        project = self.store.get_or_create_project(name)

        summary = summarize_turns(self.summarizer, turns)
        if not summary.strip():
            return None
        embedding = self.embedder.embed(summary)

        sub = None
        if external_id is not None:
            sub = self.store.get_sub_conversation_by_external_id(project.id, external_id)
        if sub is None:
            sub = self.store.start_sub_conversation(project.id, external_id=external_id)
            self.tracker.log_session_end()  # count distinct sessions, not re-ingests

        self.store.replace_memory_chunks_for_sub_conversation(project.id, sub.id, summary, embedding)
        self.store.update_sub_conversation(
            sub.id,
            raw_turns=json.dumps(turns),
            summary=summary,
            ended_at=datetime.now(timezone.utc),
        )
        return summary

    def stats(self) -> dict:
        return self.tracker.stats()
