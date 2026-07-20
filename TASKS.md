
# TokenSense — Task Sheet

Tracks remaining work against `tokensense_project.md`. Update as items land or scope changes.

## Done (initial iteration)

- [x] Package scaffold — `pyproject.toml`, repo layout, `.gitignore`, `README.md`
- [x] Memory layer — `memory/store.py` (pgvector models: projects, sub-conversations, memory chunks, documents), `embedder.py`, `retriever.py`, `documents.py`
- [x] Cache decision layer — `cache/decision.py`
- [x] Summarizers — shared `BaseSummarizer` + structured extraction prompt; `OllamaSummarizer` (local default) and `LiteLLMSummarizer` (API fallback); sliding window and session-end reuse the same incremental chain
- [x] Provider registry — thin Anthropic/OpenAI/Gemini/Ollama wrappers over LiteLLM
- [x] Middleware — `SlidingWindow` + `build_payload`
- [x] Tracker — token/CO₂ accounting matching `client.stats()`
- [x] `client.py` / `project.py` entry points
- [x] Unit tests for DB-independent logic (15 passing)

## Done (revised-architecture iteration — zero-code server, per `tokensense_project.md`)

- [x] Phase 0 — multi-project schema: denormalized `project_id` on both chunk tables, `ON DELETE CASCADE` throughout, retrieval statements that structurally require `project_id`, btree + HNSW indexes, migration script (`scripts/migrate_add_project_id.py`)
- [x] Phase 1 — proxy transport: `server/proxy.py` (OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`, passthrough body, per-request key forwarding, `X-TokenSense-*` headers), `server/config.py`, tracker wired identically to SDK path
- [x] Phase 2 — MCP transport: `server/mcp_server.py` (`get_project_context`, `end_session`, `switch_project`), `tokensense serve --mcp` CLI, Claude Code config snippet in README
- [x] Phase 3 — project resolution: `server/project_resolve.py` chain + thread-safe session bindings, concurrency isolation tests
- [x] Phase 4 — session lifecycle: idle-timeout sweep, atexit flush, explicit-only `switch_project`, env-change guard
- [x] Phase 5 — cross-project documents: per-project duplication decided (`docs/decisions.md`) + shared-file test
- [x] Phase 6 — benchmark harness: `benchmarks/run.py` (direct vs. proxy transport comparison, tracker-parity check) + CI parity test

## Next up

### 1. Local dev environment & integration testing
- [x] Stand up Postgres + pgvector (Docker) for local dev — `docker-compose.yml` (pgvector/pgvector:pg17,
      `postgresql://tokensense:tokensense@localhost:5432/tokensense`); verified live: schema auto-creation,
      HNSW indexes, project-isolated top-k retrieval, cascade delete
- [x] End-to-end smoke test: create project → chat → `end_session()` → confirm the memory chunk is retrievable in a second session — `tests/test_live_integration.py` (real Store/Retriever/SlidingWindow against Docker Postgres, deterministic fakes at the network boundary; auto-skips when the DB is down; `TOKENSENSE_TEST_DB_URL` overrides the URL for CI)
- [x] Point at a local Ollama instance for summarization testing — `tests/test_live_ollama.py` (opt-in via
      `TOKENSENSE_LIVE_OLLAMA=1`; real phi3:mini summaries persisted and retrieved end-to-end). Fixed a
      latent bug found by this: the default model name was `phi3-mini`, which is not a valid Ollama tag —
      corrected to `phi3:mini` in client.py, server/config.py, summarizers/ollama.py
- [x] Summarization latency follow-up — `SUMMARY_MAX_TOKENS=512` cap added to both summarizers;
      `benchmarks/summarizer_models.py` compared phi3:mini / qwen2.5:3b / llama3.2:3b on latency +
      fact retention: qwen2.5:3b won both (2.0s avg, 7/7 facts) and is now the default
      (`docs/decisions.md` #5). Live E2E run went from 6m22s to 9.7s
- [ ] Integration tests against a live DB — the two live test modules now cover the SDK path (store,
      retrieval, isolation, cascade, session lifecycle); server transports (proxy/MCP) still only tested
      against mocked upstreams

### 2. Embedding model follow-up
- [x] Local-first embedding default: `ollama/nomic-embed-text` (768-dim); OpenAI `text-embedding-3-*`
      still supported via their `dimensions` param so one schema fits all models (`docs/decisions.md` #4)
- [x] `EMBEDDING_DIM` now 768 to match the default; `Store` verifies the live schema's vector dimension
      at startup and raises with re-migration instructions on mismatch (verified live against the old
      1536 schema). Re-migration story documented in `docs/decisions.md` #4

### 3. Cache decision layer wiring
- [x] `choose_strategy` wired into `Project.chat`'s document path: session-pinned + live-TTL documents ride
      verbatim (Anthropic `cache_control` markers; plain text for auto-caching providers), everything else
      falls back to RAG chunks with verbatim docs excluded from retrieval (`docs/decisions.md` #6).
      New `Project.add_document()`, provider caching flags, `documents.content` column
- [x] `provider_ttl_expires_at` set via `Store.mark_document_used(cache_write=True, ...)` after the first
      successful completion carrying the document; refreshed on each NATIVE_CACHE send. Covered by
      `tests/test_cache_wiring.py` (6 tests) + a live-DB lifecycle test in `tests/test_live_integration.py`
- [ ] Cache wiring is SDK-only for now — the server transports have no document-ingestion surface at all
      yet (`ServerEngine` predates `add_document`); revisit when documents get a server story

### 3.5 Claude Code integration (primary product surface — MCP memory layer for Claude Code)

Reframed 2026-07-18: the MCP-on-Claude-Code memory layer is the product; SDK/proxy/demo are secondary.

- [x] Deterministic session capture — `tokensense ingest-transcript` (`server/transcript.py` parser +
      `ServerEngine.ingest_transcript_turns`): a Claude Code `SessionEnd` hook pipes the transcript in;
      parsing keeps conversation text and drops thinking/tool/sidechain noise; ingestion is idempotent
      per session (`sub_conversations.external_id`); hook always exits 0. Verified live against a real
      transcript of this repo (summarized by qwen, embedded by nomic, retrievable). Hook setup in README
- [x] `save_context(note)` MCP tool — `ServerEngine.save_context` embeds + stores the note as an
      immediately retrievable memory chunk under the session's sub-conversation (survives unclean
      session ends; deliberately not window-folded, so a later end_session doesn't duplicate it)
- [x] MCP-side savings accounting — `get_project_context` now logs retrieved-summary tokens vs. the
      stored `raw_turns` tokens of the sessions each summary condenses (`docs/decisions.md` #7:
      measured baseline, not modeled; note-only chunks and document chunks log nothing), via a new
      project-scoped `memory_chunk_topk_with_sources_stmt` + `Retriever.retrieve_with_sources`.
      New `stats` MCP tool exposes the tracker; scope is per server process (= one Claude Code
      session) — persisted lifetime counters deliberately deferred
- [x] Dogfood: `.mcp.json` + SessionEnd hook (`.claude/settings.json`) added to this repo with absolute
      venv paths; hook command verified verbatim against a real SessionEnd payload (exit 0, idempotent).
      `tests/test_live_mcp_stdio.py` runs the Phase 2 exit criterion for real: ingest-captured memory →
      fresh `tokensense serve --mcp` subprocess → MCP stdio handshake → `get_project_context` returns
      the prior session's decision (passes in ~10s under `TOKENSENSE_LIVE_OLLAMA=1`)
- [x] `PreCompact` hook as a second capture point — same `tokensense ingest-transcript` command
      (both hooks deliver {session_id, transcript_path, cwd}; per-session idempotency means the
      SessionEnd capture supersedes the PreCompact snapshot). Added to README + this repo's
      `.claude/settings.json` dogfood config

### 4. Demo UI (FastAPI + React)
- [ ] Backend endpoints: create project, chat, end_session, list documents, stats
- [ ] Frontend: chat view, live savings panel, retrieved-context transparency panel, provider/model switcher
- [ ] Hosting decision for the "no install required" portfolio demo

### 5. Benchmark harness
- [ ] Assemble N multi-session test scenarios
- [ ] Eval runner: full-history path vs. TokenSense-compressed path
- [ ] Pick an LLM judge model distinct from whatever's under test (avoid self-preference bias) + scoring rubric
- [ ] Publish results, including cases where compression hurts quality

### 6. Packaging & polish
- [ ] CO₂ methodology write-up (cite sources; net out the summarization step's own token/compute cost)
- [x] PyPI packaging metadata (license, classifiers, author info) — `pyproject.toml` +
      `LICENSE` (MIT); not yet published to PyPI, install is via `uvx`/`pip` from GitHub
      (`github.com/AayushKumbhare/tokensense`)
- [ ] CI: lint + pytest on push
- [x] Production Dockerfile (multi-stage, non-root) + `docker-compose.yml` `server`
      profile for a fully containerized self-host; hardened compose (restart policy,
      configurable creds)
- [x] Cloud-hosted Postgres documented as the recommended default (Neon, free tier,
      native pgvector/HNSW) so other users skip local Docker entirely — see README
      "Database"; self-hosted Docker Postgres kept as the alternative

## Phase 2 (deferred — not started, not blocking Phase 1)

A hosted consumer chat app built on top of the SDK, per the staged-audience decision in
`tokensense_project.md` (Roadmap section). Different audience, different product:

- [ ] Managed/proxied API key model (BYOK doesn't work for non-technical users)
- [ ] Hosting + billing infrastructure
- [ ] Consumer-facing UX: transparency/evidence panel with 👍/👎 feedback, topic-drift
      "move to a new tab?" nudge, before/after payload diff per turn
- [ ] Positioning against Claude Projects / ChatGPT Projects, not just Phantm/LiteLLM

Do not let these pull SDK design decisions in Phase 1 toward consumer-UX concerns.
