# TokenSense — Revised Architecture Implementation Plan

## Context

The original design required an application developer to write code against `TokenSenseClient` to get any benefit. This plan adds a zero-code path — a local server exposing both an OpenAI/Anthropic-compatible proxy and an MCP server — on top of the existing SDK, without touching the SDK's public interface. The SDK and the server are two transports into the same core engine (`middleware.py`, `memory/`, `tracker.py`).

Everything here is additive to the existing repo structure in the project doc, not a rewrite.

---

## Phase 0 — Multi-project schema (foundation, blocks everything else)

**Goal:** enforce project isolation at the query level, not by convention.

- [x] Add `project_id` as a required, indexed column on `memory_chunks` and `document_chunks` (denormalized, not just reachable via join)
- [x] `ON DELETE CASCADE` from `projects` down through `sub_conversations` → `memory_chunks`, and `documents` → `document_chunks`
- [x] Retrieval query in `retriever.py` always includes `WHERE project_id = :project_id` — no code path may omit it
- [x] Separate, explicitly-named `retrieve_cross_project(...)` function only if/when cross-project search is ever wanted — never the default
- [x] Decide index strategy: composite index (`project_id` + HNSW) now; defer partitioning (`PARTITION BY LIST (project_id)`) until benchmark harness shows single-index degradation
- [x] Migration script for existing single-project installs (if any test data already exists)

**Exit criteria:** a retrieval query for Project A returns zero rows from Project B even when a Project B chunk would score higher semantically than anything in Project A.

---

## Phase 1 — `server/` module: proxy transport

**Goal:** any tool with a `base_url` override gets memory with no code changes.

- [x] `server/proxy.py` — HTTP server implementing OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages`
- [x] Request handling: intercept → resolve project → embed current message → retrieve top-K → construct payload → forward to real provider using the user's forwarded API key → return response untouched (decide envelope-vs-passthrough per earlier open question)
- [x] API key handling: forwarded per-request, never persisted to disk or logged
- [x] `server/config.py` — port, default provider, default project fallback name
- [x] Wire proxy calls through the existing `tracker.py` so token/CO₂ logging is identical regardless of transport

**Exit criteria:** `curl` against `localhost:8317/v1/messages` with no TokenSense-specific code on the client side returns a memory-augmented response and logs savings.

---

## Phase 2 — `server/` module: MCP transport

**Goal:** Claude Code / Cursor-style tools get memory via one config block, with retrieval exposed as callable tools rather than an invisible hop.

- [x] `server/mcp_server.py` — exposes `get_project_context`, `end_session`, `switch_project` as MCP tools
- [x] `tokensense serve --mcp` CLI flag to launch this mode (`server/cli.py`)
- [x] Decide retrieval trigger policy: unconditional per-turn (simple, matches proxy behavior) vs. agent-invoked-when-relevant (more efficient, depends on host tool cooperating) — pick one as v1 default, document the tradeoff
- [x] Sample MCP config snippets for at least one real host tool (Claude Code) in README

**Exit criteria:** a fresh Claude Code session in a project directory, with only the MCP config added, correctly answers a question referencing a prior session's decision.

---

## Phase 3 — Project resolution

**Goal:** every incoming request maps unambiguously to a `project_id` before retrieval fires.

- [x] `server/project_resolve.py` implementing the resolution chain: header override → session binding → env var → cwd inference → default fallback
- [x] `session_bindings` in-memory map: `session_id → project_id`, resolved once at session start, reused for all subsequent turns in that session
- [x] `get_or_create_project(name)` — creates a `projects` row on first use, no manual project-creation step required
- [x] `infer_from_cwd()` — git repo name if available, else folder name, else nothing (falls through to default)
- [x] Concurrency test: two simultaneous sessions (different `TOKENSENSE_PROJECT` env vars) against one running server never cross-contaminate context

**Exit criteria:** two terminals open at once, different projects, same server process — each gets only its own project's memory, verified under concurrent load.

---

## Phase 4 — Session lifecycle & explicit project switching

**Goal:** mid-session project switching is deliberate, never silent.

- [x] `end_session()` auto-fires on MCP subprocess exit / proxy idle timeout (no explicit call required from the host tool)
- [x] `switch_project(name)` MCP tool: calls `end_session()` on the current binding first, then re-resolves and binds the session to the new project
- [x] Guard against a live session being silently retargeted by an env var change mid-conversation (session binding is cached at session start, not re-read per turn)

**Exit criteria:** attempting to switch projects without an explicit `switch_project` call has no effect — the session stays bound to its original project until it ends.

---

## Phase 5 — Documents crossing project boundaries

**Goal:** decide and implement how a file legitimately shared across two projects (e.g. a shared API spec) is handled given `file_hash` dedup pulls against project-scoping.

- [x] Decide: store one `documents` row per project even for identical `file_hash` (simple, no cross-project leakage risk, some storage duplication) vs. a `document_project_links` join table allowing one physical `documents`/`document_chunks` set to be referenced by multiple `project_id`s (less duplication, more complexity)
- [x] Whichever is chosen, confirm it doesn't weaken the Phase 0 isolation guarantee for `memory_chunks` (session summaries should almost certainly never be shared cross-project even if documents sometimes are)

**Exit criteria:** documented decision + schema reflecting it, with a test case for the shared-file scenario.

---

## Phase 6 — Benchmark harness updates

**Goal:** confirm the zero-code transports produce the same quality/compression tradeoff as the SDK path, since they share the same core engine but are new entry points.

- [x] Extend existing benchmark scenarios to run through the proxy and MCP transports, not just `TokenSenseClient` directly
- [x] Confirm token/CO₂ tracker numbers match between SDK and server paths for equivalent scenarios (sanity check that the shared core is actually shared, not silently diverging)

**Exit criteria:** benchmark report includes a transport-comparison table alongside the existing compression-quality results.

---

## Suggested build order

1. Phase 0 (schema) — everything downstream depends on this
2. Phase 3 (project resolution) — needed before either transport can be meaningfully tested
3. Phase 1 (proxy) — simplest transport, validates the end-to-end flow first
4. Phase 2 (MCP) — builds on proxy learnings, adds tool-call surface
5. Phase 4 (session lifecycle / switching) — layers on top of both transports
6. Phase 5 (documents) — can be deferred; not blocking for a working memory demo
7. Phase 6 (benchmarks) — ongoing, but a real pass belongs after Phases 1–4 are stable

## Open decisions to resolve before/during Phase 1

All three resolved — full rationale in `docs/decisions.md`:

- Proxy response: **passthrough** (envelope would break provider SDKs); retrieval transparency via `X-TokenSense-*` response headers
- MCP retrieval trigger: **agent-invoked** (`get_project_context` tool); unconditional per-turn retrieval is the proxy transport's job
- Documents: **per-project duplication** (join table would reintroduce cross-project reachability that Phase 0 exists to prevent)

## Status (2026-07-10) — implemented

All phases built and tested (52 tests passing, up from 15). Implementation notes:

- Phase 0 index strategy: btree on `project_id` + single-column HNSW on `embedding` (pgvector HNSW cannot be composite); `PARTITION BY LIST` deferred per plan. Migration for pre-existing installs: `scripts/migrate_add_project_id.py`.
- The shared core behind both transports is `tokensense/server/engine.py` (`ServerEngine`), reusing the SDK's `Store`/`Retriever`/`SlidingWindow`/`build_payload`/`Tracker` directly.
- Proxy limitation: `stream: true` returns a clear 400 in v1 (SSE delta assembly deferred).
- Phase 6: the harness (`benchmarks/run.py`) compares the direct engine path vs. the proxy, with a tracker-parity check (also enforced in CI by `tests/test_transport_parity.py`). The MCP transport is excluded from the scripted comparison because its retrieval is agent-invoked (no per-turn payloads to compare); its behavior is covered by `tests/test_mcp_server.py`.
- Because the host tool owns the conversation on the MCP transport, session content reaches memory via the optional `summary` argument to `end_session`.