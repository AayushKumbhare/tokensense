"""TokenSenseClient — main entry point (see project doc: Package API)."""
from __future__ import annotations

from .memory.documents import DocumentIngestor
from .memory.embedder import Embedder
from .memory.retriever import Retriever
from .memory.store import Store
from .project import Project
from .providers import get_provider
from .summarizers.litellm import LiteLLMSummarizer
from .summarizers.ollama import OllamaSummarizer
from .tracker import Tracker

# Bare names (no "ollama/" prefix) that still route to the local Ollama summarizer.
# These must be valid Ollama registry tags; "phi3:mini" is the tag for the mini
# variant ("phi3-mini" is not a real tag and fails against a live Ollama).
LOCAL_SUMMARIZER_MODELS = {"phi3:mini", "phi3", "llama3.2", "llama3.2:3b", "qwen2.5:3b"}


def build_summarizer(summarization_model: str, api_key: str | None = None):
    """Shared by the SDK client and the server transports so both paths use
    the exact same summarizer selection."""
    if summarization_model.startswith("ollama/") or summarization_model in LOCAL_SUMMARIZER_MODELS:
        model_name = summarization_model.split("/", 1)[-1]
        return OllamaSummarizer(model=model_name)
    return LiteLLMSummarizer(model=summarization_model, api_key=api_key)


class TokenSenseClient:
    def __init__(
        self,
        provider: str,
        api_key: str | None,
        db_url: str,
        model: str | None = None,
        summarization_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        window_size: int = 5,
        top_k: int = 5,
    ):
        self.provider_config = get_provider(provider)
        self.api_key = api_key
        self.window_size = window_size

        self.chat_model = self.provider_config.to_litellm_model(model or self.provider_config.DEFAULT_MODEL)

        self.store = Store(db_url)
        self.embedder = Embedder(model=embedding_model)
        self.retriever = Retriever(self.store, self.embedder, top_k=top_k)
        self.tracker = Tracker()
        self.document_ingestor = DocumentIngestor(self.store, self.embedder)

        self.summarizer = build_summarizer(summarization_model, api_key=self.api_key)
        self._projects: dict[str, Project] = {}

    def project(self, name: str) -> Project:
        if name not in self._projects:
            self._projects[name] = Project(
                name,
                store=self.store,
                retriever=self.retriever,
                summarizer=self.summarizer,
                chat_model=self.chat_model,
                api_key=self.api_key,
                window_size=self.window_size,
                tracker=self.tracker,
                provider_config=self.provider_config,
                document_ingestor=self.document_ingestor,
            )
        return self._projects[name]

    def stats(self) -> dict:
        return self.tracker.stats()
