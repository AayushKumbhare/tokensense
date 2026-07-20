# TokenSense

Model-agnostic context management middleware for LLM-powered workflows. Gives projects
persistent, selective memory across sub-conversations using RAG over embedded summaries,
instead of re-explaining prior work or stuffing full history into the context window.

See `tokensense_project.md` for the full design doc.

## Install (editable, for development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires a Postgres instance with the `pgvector` extension available (the client will
create the extension and tables on first connect). For local dev, the included
compose file runs one:

```bash
docker compose up -d --wait
```

This exposes Postgres 17 + pgvector on `localhost:5432` with a persistent volume;
the connection URL is `postgresql://tokensense:tokensense@localhost:5432/tokensense`.

Summarization and embeddings default to local Ollama models (no conversation content
leaves your machine), so the default configuration also needs:

```bash
ollama pull qwen2.5:3b        # summarizer (see docs/decisions.md #5)
ollama pull nomic-embed-text  # embeddings, 768-dim (see docs/decisions.md #4)
```

API models remain available via `summarization_model=` / `embedding_model=` (or the
`TOKENSENSE_SUMMARIZATION_MODEL` / `TOKENSENSE_EMBEDDING_MODEL` env vars for the
server). Changing the *embedding* model invalidates stored vectors — see the
re-migration note in `docs/decisions.md`.

## Quickstart

```python
from tokensense import TokenSenseClient

client = TokenSenseClient(
    provider="anthropic",
    api_key="your-key-here",
    db_url="postgresql://localhost/tokensense",
)

project = client.project("react-dashboard-build")
project.add_document("api_spec.md")  # optional: pin a file into the session
response = project.chat(messages=[{"role": "user", "content": "Let's work on the auth flow today"}])
print(client.stats())

project.end_session()
```

Documents added to a session ride along verbatim using the provider's prompt cache
while its TTL is live (cheap cached reads); once it expires — or on providers without
prompt caching — they're served as retrieved chunks instead (`docs/decisions.md` #6).

## Zero-code server (proxy + MCP)

The SDK above requires writing code against `TokenSenseClient`. The server exposes the
same core engine through two zero-code transports (install with `pip install -e ".[server]"`).

### Proxy transport

Any tool with a `base_url` override gets memory with no code changes:

```bash
export TOKENSENSE_DB_URL=postgresql://localhost/tokensense
tokensense serve            # listens on localhost:8317
```

Point your existing client at it — your API key is forwarded per-request, never stored:

```bash
curl localhost:8317/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-tokensense-project: my-project" \
  -d '{"model": "claude-sonnet-5", "max_tokens": 1024,
       "messages": [{"role": "user", "content": "What did we decide last session?"}]}'
```

`/v1/chat/completions` (OpenAI-compatible) works the same way with an `Authorization`
header. The response body is a pure passthrough; retrieval transparency comes back in
`X-TokenSense-*` response headers. Project resolution: `X-TokenSense-Project` header →
existing session binding → `TOKENSENSE_PROJECT` env var → git repo / folder name → `default`.
Streaming is not supported yet.

### MCP transport

For Claude Code, Cursor, and similar tools — memory is exposed as callable tools:
`get_project_context` (retrieval), `save_context` (persist a decision mid-session,
without waiting for session end), `end_session`, `switch_project`, and `stats`
(token/CO₂ savings: retrieved-summary tokens vs. the raw session tokens they
replaced, per `docs/decisions.md` #7; scoped to the current server process, i.e.
one Claude Code session). Add to `.mcp.json` in your project directory (Claude Code):

```json
{
  "mcpServers": {
    "tokensense": {
      "command": "tokensense",
      "args": ["serve", "--mcp"],
      "env": { "TOKENSENSE_DB_URL": "postgresql://localhost/tokensense" }
    }
  }
}
```

The project binds once at session start (from cwd/git inference or `TOKENSENSE_PROJECT`)
and never silently changes; switching mid-session requires the explicit `switch_project`
tool.

### Session capture (Claude Code hooks)

The host tool owns the conversation on the MCP transport, so relying on the agent to
call `end_session` with notes makes memory capture best-effort. For deterministic
capture, add hooks to `.claude/settings.json` — Claude Code pipes the session's
transcript to TokenSense, which summarizes it (local qwen by default), embeds it,
and stores it as project memory. `SessionEnd` is the primary capture point;
`PreCompact` fires the same command right before Claude Code compacts its context —
context exhaustion is exactly when memory matters most, and this snapshots the
session before compaction rewrites it:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "TOKENSENSE_DB_URL=postgresql://tokensense:tokensense@localhost:5432/tokensense tokensense ingest-transcript"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "TOKENSENSE_DB_URL=postgresql://tokensense:tokensense@localhost:5432/tokensense tokensense ingest-transcript"
          }
        ]
      }
    ]
  }
}
```

Ingestion is idempotent per Claude Code session (re-firing either hook, or re-ingesting a
resumed session's grown transcript, updates that session's memory in place — so a
PreCompact snapshot is simply superseded by the SessionEnd capture), keeps only
conversation text (thinking blocks, tool calls/outputs, and subagent sidechains are
dropped), and always exits 0 so a capture failure never breaks the host tool. The same
command works standalone: `tokensense ingest-transcript <path-to-transcript.jsonl>`.

See `docs/decisions.md` for the resolved design decisions (passthrough vs. envelope,
agent-invoked retrieval, per-project document duplication).

## Benchmarks

```bash
python benchmarks/run.py --offline                                  # plumbing + transport parity, no services
python benchmarks/run.py --db-url postgresql://localhost/ts_bench   # real embeddings/summarizer
```

Prints a transport-comparison table (direct engine path vs. proxy) with token savings and
probe recall, plus a tracker-parity check confirming both transports share the same core.

## Tests

```bash
pytest
```

When the Docker database is up, `tests/test_live_integration.py` additionally runs an
end-to-end smoke test against it (session memory surviving into a second session, and
project isolation on real HNSW-ranked retrieval); it skips itself automatically when
the database is unreachable. Set `TOKENSENSE_TEST_DB_URL` to point it elsewhere.

With `TOKENSENSE_LIVE_OLLAMA=1` and a local Ollama serving the default models
(`qwen2.5:3b` + `nomic-embed-text`), `tests/test_live_ollama.py` also runs the fully
local stack — real summarization and real embeddings, live semantic ranking, no API
keys. `benchmarks/summarizer_models.py` compares candidate local summarizer models on
latency and fact retention.

The remaining tests cover the logic that doesn't require a live Postgres connection: sliding window,
summarizer chaining, cache-vs-RAG decision, tracker math, document chunking, project
isolation guards (compiled query scoping, cascade FKs), the resolution chain, session
lifecycle under concurrency, both server transports (with mocked upstreams), and
SDK-vs-proxy transport parity. Migrating a pre-multi-project install:
`python scripts/migrate_add_project_id.py <db-url>`.
