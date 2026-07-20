# TokenSense — Resolved Design Decisions

Decisions from the "open decisions" list in `tokensense_project.md`, with rationale.

## 1. Proxy response: passthrough (not an envelope)

**Decision:** the proxy returns the provider's response body byte-for-byte
untouched. Retrieval transparency is exposed via response headers instead:

- `X-TokenSense-Project` — the project the request was scoped to
- `X-TokenSense-Session` — the session id used
- `X-TokenSense-Retrieved` — number of memory/document chunks injected
- `X-TokenSense-Tokens-Saved` — cumulative tokens saved this server run

**Why:** the whole point of the proxy is zero-code adoption. Client SDKs
(openai, anthropic) parse the response body strictly; any envelope would break
them. Headers are invisible to SDKs that don't look for them and available to
tools that do.

## 2. MCP retrieval trigger: agent-invoked

**Decision:** retrieval on the MCP transport fires only when the host agent
calls `get_project_context`. There is no unconditional per-turn injection.

**Why:** MCP tools are by construction agent-invoked; the host tool owns the
conversation and TokenSense never sees turns it isn't shown. Agent-invoked is
more token-efficient (no retrieval on turns that don't need memory) but
depends on the host actually calling the tool — the tool descriptions are
written to make that reliable. Users who want unconditional per-turn retrieval
should use the proxy transport, which sits in the request path and can
guarantee it. This tradeoff is the documented difference between the two
transports, not a bug in either.

Corollary: because the host owns the conversation, session content reaches
project memory through the optional `summary` argument to `end_session`.

## 3. Documents crossing project boundaries: per-project duplication

**Decision:** one `documents` row (and one set of `document_chunks`) per
project, even when the same physical file (identical `file_hash`) is ingested
into several projects. `file_hash` dedup is scoped *within* a project:
re-ingesting an unchanged file into the same project is a no-op; ingesting it
into a second project stores a second copy.

**Why:** a `document_project_links` join table would save storage but
reintroduce exactly the cross-project reachability Phase 0 exists to prevent —
every retrieval and deletion path would need link-aware logic, and a bug there
leaks content across projects. Duplication keeps the Phase 0 invariant simple:
every chunk row has exactly one `project_id`, `ON DELETE CASCADE` works
per-project with no reference counting, and retrieval never needs a join.
Storage cost is bounded (documents are chunked text + embeddings) and
irrelevant at current scale.

**Explicitly unchanged:** `memory_chunks` (session summaries) are never shared
cross-project under any future design; only documents were ever candidates.

## 4. Embeddings: local-first default at a single fixed dimension (768)

**Decision:** the default embedding model is `ollama/nomic-embed-text` (768
dimensions) and `EMBEDDING_DIM` is 768. OpenAI `text-embedding-3-*` models
remain supported: they accept a `dimensions` request parameter, so the
`Embedder` asks them for 768-dim vectors and every supported model fits the
same schema. `Store` verifies the existing tables' vector dimension at startup
and refuses to run against a mismatched schema.

**Why:** the summarizer already defaults to a local Ollama model; defaulting
embeddings to OpenAI meant Ollama-only users silently sent every message and
summary to a hosted API — the opposite of the local-first posture, and a
surprise data-egress path. Truncated 768-dim `text-embedding-3-small` vectors
lose little retrieval quality (the model is trained for Matryoshka-style
truncation), which is what makes a single fixed dimension viable.

**Re-migration story:** embeddings from different models are not comparable,
so there is no in-place migration between embedding models — even same-dim
swaps invalidate stored vectors. Changing models means: drop the five tables
and re-ingest/re-embed (raw turns are retained on `sub_conversations`, so
summaries could be re-embedded programmatically later if that ever matters).
The startup guard turns the silent-corruption case into a clear error.

## 5. Summarizer default: qwen2.5:3b, output-capped

**Decision:** the local summarizer default is `qwen2.5:3b` with output capped
at `SUMMARY_MAX_TOKENS` (512). Both `OllamaSummarizer` and `LiteLLMSummarizer`
apply the cap.

**Why:** uncapped, phi3:mini generated until it filled its context — 1–2
minutes per summarize call, paid on every `end_session()`. With the cap,
`benchmarks/summarizer_models.py` (3 timed runs per model, fixed dev-session
transcript, fact-retention checklist) measured: qwen2.5:3b 2.0s avg with 7/7
facts retained on every run; llama3.2:3b 3.1s, 6/7; phi3:mini 4.8s, 6/7.
qwen2.5:3b won on both axes. Summaries are meant to be dense structured
extractions, so a hard cap is consistent with the design, not a compromise.

## 6. Document caching: session-pinned writes, TTL-gated reads, RAG fallback

**Decision:** `choose_strategy` (cache/decision.py) is wired into
`Project.chat`'s document path as follows:

- **Cache write trigger:** `Project.add_document()` pins the document to the
  current session. On a caching provider, pinned documents enter every payload
  of that session verbatim; when the provider call succeeds,
  `Store.mark_document_used(cache_write=True, ttl_expires_at=now + provider
  TTL)` records the write. This answers "where does `provider_ttl_expires_at`
  get set": after the first successful completion that carried the document.
- **NATIVE_CACHE:** documents with a live TTL keep riding verbatim in later
  sessions, and each successful send refreshes the TTL (matching Anthropic's
  refresh-on-read semantics).
- **RAG_RETRIEVAL:** expired-TTL documents, never-cached documents outside
  their add-session, and all documents on non-caching providers (Ollama) are
  served as top-K retrieved chunks. Retrieval stamps `last_used_at` but never
  touches the TTL.
- Verbatim documents are excluded from chunk retrieval
  (`exclude_document_ids` on the still-project-scoped statement builder) so
  content is never sent twice; they lead the payload in stable order because
  provider caches key on identical prefixes. Providers declare
  `SUPPORTS_PROMPT_CACHING` / `PROMPT_CACHE_TTL_SECONDS` /
  `NEEDS_CACHE_MARKERS` (Anthropic gets explicit `cache_control` blocks;
  OpenAI/Gemini prefix-cache automatically).

**Why session-pinning as the write trigger:** `choose_strategy` returns RAG
for never-cached documents, so something outside it must decide when a cache
is first worth writing. Explicitly adding a file to a live session is the
strongest available signal that its full content is about to matter; ambient
cross-session reuse stays on the cheap RAG path, which is the documents
module's stated purpose. `documents.content` (full original text) was added
to make identical-prefix resends possible at all.

## 7. MCP savings accounting: summary tokens vs. the raw session they replace

**Decision:** on the MCP transport, each `get_project_context` call logs to
the tracker: *actual* = tokens of the retrieved memory-chunk summaries,
*baseline* = tokens of the stored `raw_turns` of the sub-conversations those
summaries condense. A `stats` MCP tool exposes the tracker. Counting uses
litellm's default tokenizer (there is no host model to count against; the
same approximation applies to both sides of the subtraction, so the delta is
fair). Chunks whose source session has no stored raw turns (e.g. `save_context`
notes) and document chunks log nothing — they replace no session.

**Why this counterfactual:** the SDK/proxy tracker compares the compressed
payload against full-history resend, but on MCP the host tool owns the
conversation and TokenSense never sees the payload. What it *can* measure is
what the summary stands in for: without TokenSense, giving the agent that
prior-session context means re-pasting or re-explaining the raw session. That
is exactly the raw_turns the transcript-ingestion path already persists, so
the baseline is measured, not modeled.

**Scope:** the tracker is in-memory, so `stats` covers the current server
process — one Claude Code session on the MCP transport (one subprocess per
session). Cumulative cross-session accounting would need persisted counters;
deferred until the pitch needs a lifetime number rather than a per-session one.

## 8. Default summarizer/embedder: OpenAI, not Ollama (reverses #4/#5's default)

**Decision:** `ServerConfig` and `TokenSenseClient` now default
`summarization_model` to `gpt-4o-mini` and `embedding_model` to
`text-embedding-3-small`, not the local Ollama models decisions #4 and #5
picked as defaults. Ollama remains fully supported and unchanged — set both
`TOKENSENSE_SUMMARIZATION_MODEL=qwen2.5:3b` and
`TOKENSENSE_EMBEDDING_MODEL=ollama/nomic-embed-text` to opt back in.

**Why:** #4/#5 optimized for a single local, cost-free, privacy-preserving
install (the project owner's own machine). Once outside collaborators started
using TokenSense against a shared or per-user hosted database (see
`scripts/create_user_database.py`), the local-first default became a setup
tax on every new user: install Ollama, pull ~4GB across two models, before
anything works. One OpenAI key — which most Claude Code users already have
some equivalent of — covers both summarization and embeddings, so the
collaborator's entire setup is `.mcp.json` plus that key. This is explicitly
a default flip, not a removal: users who want the original privacy/cost
posture keep it via the two env vars above, and `LOCAL_SUMMARIZER_MODELS`
routing in `client.py` is unchanged.
