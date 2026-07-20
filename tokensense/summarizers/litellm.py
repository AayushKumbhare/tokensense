"""API-model fallback summarizer for users without a local model available."""
from __future__ import annotations

import litellm

from .base import SUMMARY_MAX_TOKENS, BaseSummarizer


class LiteLLMSummarizer(BaseSummarizer):
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = SUMMARY_MAX_TOKENS):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def _complete(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            max_tokens=self.max_tokens,
        )
        return response["choices"][0]["message"]["content"]
