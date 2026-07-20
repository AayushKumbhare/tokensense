"""Sliding window over the active sub-conversation: recent turns stay verbatim,
older ones get folded into a running structured summary instead of being
dropped (see project doc: Architecture / Sliding Window)."""
from __future__ import annotations

from collections import deque

from .summarizers.base import BaseSummarizer


class SlidingWindow:
    def __init__(self, summarizer: BaseSummarizer, window_size: int = 5):
        self.summarizer = summarizer
        self.window_size = window_size
        self._verbatim: deque[dict] = deque()
        self.rolling_summary: str | None = None

    def add_turn(self, turn: dict) -> None:
        self._verbatim.append(turn)
        while len(self._verbatim) > self.window_size:
            evicted = self._verbatim.popleft()
            self.rolling_summary = self.summarizer.summarize(self.rolling_summary, [evicted])

    @property
    def verbatim_turns(self) -> list[dict]:
        return list(self._verbatim)

    def finalize(self) -> str:
        """Session-end summary: same summarizer, one final call anchored to the
        still-verbatim tail rather than trusting a pure chain of compressions
        (see project doc: One Summarizer, Shared Across Sliding Window and
        Session End)."""
        return self.summarizer.summarize(self.rolling_summary, self.verbatim_turns)


def build_payload(
    retrieved_chunks: list[str],
    rolling_summary: str | None,
    verbatim_turns: list[dict],
    current_message: str,
    cached_documents: list[tuple[str, str]] | None = None,
    cache_markers: bool = False,
) -> list[dict]:
    """[cached documents] + [retrieved_context] + [current conversation window
    (verbatim)] + [current message]

    `cached_documents` are (filename, full_text) pairs served via the
    NATIVE_CACHE strategy (cache/decision.py). They lead the payload in stable
    order because provider prompt caches key on an identical prefix. With
    `cache_markers`, each is emitted as a content block carrying an Anthropic
    `cache_control` marker; providers with automatic prefix caching just get
    plain text.
    """
    messages: list[dict] = []
    for filename, text in cached_documents or []:
        body = f"Project document ({filename}):\n{text}"
        if cache_markers:
            content: str | list = [
                {"type": "text", "text": body, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            content = body
        messages.append({"role": "system", "content": content})
    if retrieved_chunks:
        context_block = "\n\n".join(retrieved_chunks)
        messages.append(
            {"role": "system", "content": f"Relevant context from past sessions in this project:\n{context_block}"}
        )
    if rolling_summary:
        messages.append(
            {"role": "system", "content": f"Summary of earlier turns in this session:\n{rolling_summary}"}
        )
    messages.extend(verbatim_turns)
    messages.append({"role": "user", "content": current_message})
    return messages
