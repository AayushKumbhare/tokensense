"""Structured-extraction summarizer shared by the in-session sliding window and
the end-of-session memory chunk (see project doc: One Summarizer, Shared
Across Sliding Window and Session End)."""
from __future__ import annotations

from abc import ABC, abstractmethod

# Hard cap on summary output length. Uncapped, small local models can generate
# until they fill their context window — phi3:mini measured 1–2 minutes per
# summarize call on an M-series Mac. Summaries are meant to be dense; capping
# bounds end_session latency without losing the structured extraction.
SUMMARY_MAX_TOKENS = 512

EXTRACTION_PROMPT = """\
Extract and preserve from this conversation:
- All specific facts, decisions, and conclusions reached
- All names, numbers, file names, or identifiers mentioned
- All open questions or unresolved items
- The overall goal and current progress

Compress everything else. Output as dense structured text.
"""


class BaseSummarizer(ABC):
    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Call the underlying model and return its raw text response."""

    def summarize(self, prior_summary: str | None, new_turns: list[dict]) -> str:
        turns_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in new_turns)
        sections = [EXTRACTION_PROMPT]
        if prior_summary:
            sections.append(f"Existing summary so far:\n{prior_summary}")
        sections.append(f"New conversation turns to fold in:\n{turns_text}")
        prompt = "\n\n".join(sections)
        return self._complete(prompt)
