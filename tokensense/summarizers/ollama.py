"""Local, API-cost-free summarizer — default path (see project doc: Summarization Strategy)."""
from __future__ import annotations

import litellm

from .base import SUMMARY_MAX_TOKENS, BaseSummarizer


class OllamaSummarizer(BaseSummarizer):
    # qwen2.5:3b won the benchmark (benchmarks/summarizer_models.py) on both
    # latency and fact retention vs phi3:mini and llama3.2:3b.
    def __init__(self, model: str = "qwen2.5:3b", max_tokens: int = SUMMARY_MAX_TOKENS):
        self.model = f"ollama/{model}"
        self.max_tokens = max_tokens

    def _complete(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]
