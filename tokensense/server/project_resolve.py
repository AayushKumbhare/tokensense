"""Map every incoming request unambiguously to a project name before retrieval
fires (see revised architecture plan, Phase 3).

Resolution chain, first match wins:
    1. explicit header override (X-TokenSense-Project)
    2. existing session binding (resolved once at session start, cached)
    3. TOKENSENSE_PROJECT environment variable
    4. cwd inference (git repo name, else folder name)
    5. default fallback

The session binding is deliberately cached at session start and never re-read
per turn, so an env var change mid-conversation cannot silently retarget a
live session (Phase 4 guard). Switching requires an explicit switch_project.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path

PROJECT_HEADER = "x-tokensense-project"
SESSION_HEADER = "x-tokensense-session"
PROJECT_ENV_VAR = "TOKENSENSE_PROJECT"
DEFAULT_PROJECT_NAME = "default"


def infer_from_cwd(cwd: str | os.PathLike | None = None) -> str | None:
    """Git repo name if available, else folder name, else nothing."""
    path = Path(cwd) if cwd is not None else Path.cwd()
    try:
        path = path.resolve()
    except OSError:
        return None
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate.name
    if path.name:
        return path.name
    return None


def resolve_project_name(
    *,
    header: str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
    default: str = DEFAULT_PROJECT_NAME,
) -> str:
    """Run the resolution chain (minus session binding, which the caller owns
    via SessionBindings) and return a project name. Never returns None."""
    if header:
        return header
    env = os.environ if env is None else env
    env_project = env.get(PROJECT_ENV_VAR)
    if env_project:
        return env_project
    inferred = infer_from_cwd(cwd)
    if inferred:
        return inferred
    return default


class SessionBindings:
    """Thread-safe session_id -> project_id map. A binding is resolved once at
    session start and reused for every subsequent turn in that session."""

    def __init__(self):
        self._bindings: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> str | None:
        with self._lock:
            return self._bindings.get(session_id)

    def bind(self, session_id: str, project_id: str) -> None:
        with self._lock:
            self._bindings[session_id] = project_id

    def get_or_bind(self, session_id: str, resolve) -> str:
        """Return the existing binding, or atomically resolve and bind one.
        `resolve` is only called when no binding exists."""
        with self._lock:
            existing = self._bindings.get(session_id)
            if existing is not None:
                return existing
            project_id = resolve()
            self._bindings[session_id] = project_id
            return project_id

    def release(self, session_id: str) -> str | None:
        with self._lock:
            return self._bindings.pop(session_id, None)

    def active_sessions(self) -> list[str]:
        with self._lock:
            return list(self._bindings)
