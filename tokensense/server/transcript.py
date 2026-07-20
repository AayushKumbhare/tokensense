"""Claude Code transcript (JSONL) → conversation turns for memory ingestion.

Claude Code writes one JSONL file per session under
~/.claude/projects/<munged-cwd>/<session-id>.jsonl. Lines carry many event
types; only `user` and `assistant` lines hold conversation content, as
Anthropic-API-shaped messages whose content is either a plain string or a
list of blocks (text / thinking / tool_use / tool_result).

What we keep is what a session summary needs: what the user asked for and
what the assistant said it did. Everything mechanical — thinking blocks, tool
calls and their outputs, subagent sidechains, injected <system-reminder>
scaffolding — is noise at summary altitude and is dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

# User-turn text that is injected scaffolding, not something the user said.
_META_PREFIXES = (
    "<system-reminder",
    "<command-name",
    "<command-message",
    "<command-args",
    "<local-command",
    "<ide_selection",
    "<ide_opened_file",
    "<task-notification",
    "[SYSTEM NOTIFICATION",
    "Caveat: the messages below were generated",
)

# A single runaway turn (huge paste, dumped file) shouldn't dominate the
# summarizer's context; the head carries the intent.
MAX_TURN_CHARS = 4000


def _text_from_content(content, *, skip_meta: bool) -> str:
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = content
    else:
        return ""
    parts = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text", "").strip()
        if not text:
            continue
        if skip_meta and text.startswith(_META_PREFIXES):
            continue
        parts.append(text)
    return "\n".join(parts)


def extract_turns(transcript_path: str | Path) -> list[dict]:
    """Parse a Claude Code session transcript into [{role, content}, ...].

    Tolerant of unknown line types and malformed lines: transcripts are an
    internal Claude Code format that may evolve, and capture must degrade to
    fewer turns, never crash the SessionEnd hook.
    """
    turns: list[dict] = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("type") not in ("user", "assistant"):
                continue
            if obj.get("isSidechain"):
                continue  # subagent conversation, not the user's session
            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _text_from_content(message.get("content"), skip_meta=(role == "user"))
            if not text:
                continue
            turns.append({"role": role, "content": text[:MAX_TURN_CHARS]})
    return turns


def summarize_turns(summarizer, turns: list[dict], batch_size: int = 20) -> str:
    """Fold a full transcript through the shared incremental summarizer chain,
    batch by batch — same chain the sliding window uses, so transcript-derived
    memory matches SDK-session memory in shape."""
    summary = None
    for start in range(0, len(turns), batch_size):
        summary = summarizer.summarize(summary, turns[start : start + batch_size])
    return summary or ""
