"""HTTP proxy transport: OpenAI-compatible /v1/chat/completions and
Anthropic-compatible /v1/messages (see revised architecture plan, Phase 1).

Any tool with a base_url override gets memory with no code changes: the proxy
intercepts the request, resolves the project, retrieves top-K memory for the
current message, swaps the client's (uncompressed) history for the compressed
payload, forwards to the real provider with the caller's own API key, and
returns the provider response body untouched.

Resolved open decisions:
- Response shape: passthrough, not an envelope. Rewriting the body would break
  provider SDKs parsing the response; retrieval transparency is exposed via
  X-TokenSense-* response headers instead.
- API keys are forwarded per-request from the incoming headers and are never
  persisted or logged.
- Streaming (stream: true) is not supported in v1 and returns a clear 400;
  supporting it requires assembling assistant text from SSE deltas to feed the
  sliding window, which is deferred.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .engine import ServerEngine
from .project_resolve import PROJECT_HEADER, SESSION_HEADER

CWD_HEADER = "x-tokensense-cwd"
DEFAULT_SESSION_ID = "default"

# The only request headers forwarded upstream. Everything else (host,
# content-length, tokensense headers) is dropped; API keys pass through here
# and are never stored or logged.
FORWARDED_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "openai-organization",
    "openai-project",
)


def _message_text(content) -> str:
    """Extract plain text from a message content field that may be a string or
    a list of content blocks (both OpenAI and Anthropic use these shapes)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _last_user_message(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message.get("content"))
    return None


def _extract_assistant_text(provider: str, response_body: dict) -> str:
    if provider == "openai":
        return response_body["choices"][0]["message"]["content"] or ""
    blocks = response_body.get("content", [])
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _to_openai_body(body: dict, payload: list[dict]) -> dict:
    return {**body, "messages": payload}


def _to_anthropic_body(body: dict, payload: list[dict]) -> dict:
    """Anthropic takes system as a top-level param, not a message role."""
    system_parts = [m["content"] for m in payload if m["role"] == "system"]
    chat_messages = [m for m in payload if m["role"] != "system"]
    upstream = {**body, "messages": chat_messages}
    existing = body.get("system")
    if system_parts:
        joined = "\n\n".join(system_parts)
        if isinstance(existing, str) and existing:
            upstream["system"] = f"{existing}\n\n{joined}"
        elif isinstance(existing, list):
            upstream["system"] = existing + [{"type": "text", "text": joined}]
        else:
            upstream["system"] = joined
    return upstream


def create_app(engine: ServerEngine, *, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    owns_client = http_client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = http_client or httpx.AsyncClient(timeout=600.0)
        try:
            yield
        finally:
            engine.end_all()
            if owns_client:
                await app.state.client.aclose()

    app = FastAPI(title="TokenSense proxy", lifespan=lifespan)

    async def _handle(request: Request, provider: str, upstream_path: str) -> Response:
        body = await request.json()
        if body.get("stream"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "TokenSense proxy does not support stream=true yet; retry without streaming.",
                    }
                },
            )

        current_message = _last_user_message(body.get("messages", []))
        if current_message is None:
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "invalid_request_error", "message": "No user message found."}},
            )

        session_id = request.headers.get(SESSION_HEADER) or DEFAULT_SESSION_ID
        session = engine.get_session(
            session_id,
            project_header=request.headers.get(PROJECT_HEADER),
            cwd=request.headers.get(CWD_HEADER),
        )

        forwarded = {k: v for k, v in request.headers.items() if k.lower() in FORWARDED_HEADERS}
        forwarded["content-type"] = "application/json"
        upstream_base = engine.config.upstreams[provider]

        with session.lock:
            payload, retrieved = engine.prepare_payload(session, current_message)
            upstream_body = (
                _to_openai_body(body, payload) if provider == "openai" else _to_anthropic_body(body, payload)
            )
            upstream_response = await request.app.state.client.post(
                f"{upstream_base}{upstream_path}", json=upstream_body, headers=forwarded
            )
            if upstream_response.status_code == 200:
                assistant_text = _extract_assistant_text(provider, upstream_response.json())
                engine.record_turn(
                    session,
                    current_message=current_message,
                    assistant_content=assistant_text,
                    payload=payload,
                    model=body.get("model", ""),
                )

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
            headers={
                "x-tokensense-project": session.project.name,
                "x-tokensense-session": session_id,
                "x-tokensense-retrieved": str(len(retrieved)),
                "x-tokensense-tokens-saved": str(engine.tracker.tokens_saved),
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _handle(request, "openai", "/v1/chat/completions")

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await _handle(request, "anthropic", "/v1/messages")

    @app.get("/stats")
    async def stats() -> dict:
        return engine.stats()

    @app.post("/sessions/{session_id}/end")
    async def end_session(session_id: str) -> dict:
        summary = engine.end_session(session_id)
        return {"session_id": session_id, "summary": summary}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app
