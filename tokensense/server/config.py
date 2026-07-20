"""Server configuration (see revised architecture plan, Phase 1)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .project_resolve import DEFAULT_PROJECT_NAME

DEFAULT_PORT = 8317

# Upstream bases the proxy forwards to, keyed by provider name.
DEFAULT_UPSTREAMS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}


@dataclass
class ServerConfig:
    db_url: str
    port: int = DEFAULT_PORT
    default_provider: str = "anthropic"
    default_project: str = DEFAULT_PROJECT_NAME
    summarization_model: str = "qwen2.5:3b"
    embedding_model: str = "ollama/nomic-embed-text"
    window_size: int = 5
    top_k: int = 5
    idle_timeout_seconds: float = 1800.0
    ensure_schema: bool = True
    upstreams: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UPSTREAMS))

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ServerConfig":
        env = os.environ if env is None else env
        db_url = env.get("TOKENSENSE_DB_URL")
        if not db_url:
            raise ValueError("TOKENSENSE_DB_URL must be set (e.g. postgresql://localhost/tokensense)")
        return cls(
            db_url=db_url,
            port=int(env.get("TOKENSENSE_PORT", DEFAULT_PORT)),
            default_provider=env.get("TOKENSENSE_DEFAULT_PROVIDER", "anthropic"),
            default_project=env.get("TOKENSENSE_DEFAULT_PROJECT", DEFAULT_PROJECT_NAME),
            summarization_model=env.get("TOKENSENSE_SUMMARIZATION_MODEL", "qwen2.5:3b"),
            embedding_model=env.get("TOKENSENSE_EMBEDDING_MODEL", "ollama/nomic-embed-text"),
            window_size=int(env.get("TOKENSENSE_WINDOW_SIZE", 5)),
            top_k=int(env.get("TOKENSENSE_TOP_K", 5)),
            idle_timeout_seconds=float(env.get("TOKENSENSE_IDLE_TIMEOUT", 1800)),
            ensure_schema=env.get("TOKENSENSE_ENSURE_SCHEMA", "1") not in ("0", "false", "False"),
        )
