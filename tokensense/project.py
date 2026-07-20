"""A single project: owns sub-conversations, the active sliding window, and
triggers end-of-session summarization + memory storage (see project doc:
The Solution / Architecture)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import litellm

from .cache.decision import ContextStrategy, choose_strategy
from .memory.documents import DocumentIngestor
from .memory.retriever import Retriever
from .memory.store import Store
from .middleware import SlidingWindow, build_payload
from .summarizers.base import BaseSummarizer
from .tracker import Tracker

DEFAULT_TOP_K = 5


class Project:
    def __init__(
        self,
        name: str,
        *,
        store: Store,
        retriever: Retriever,
        summarizer: BaseSummarizer,
        chat_model: str,
        api_key: str | None,
        window_size: int,
        tracker: Tracker,
        provider_config=None,
        document_ingestor: DocumentIngestor | None = None,
    ):
        self.name = name
        self.store = store
        self.retriever = retriever
        self.chat_model = chat_model
        self.api_key = api_key
        self.tracker = tracker
        self.provider_config = provider_config
        self.document_ingestor = document_ingestor
        self._summarizer = summarizer
        self._window_size = window_size

        self._record = store.get_or_create_project(name)
        self._sub_conversation = store.start_sub_conversation(self._record.id)
        self._window = SlidingWindow(summarizer, window_size=window_size)
        self._raw_turns: list[dict] = []
        # Documents explicitly added during this session: they are about to be
        # discussed, so they enter the payload verbatim (triggering the first
        # provider cache write) rather than waiting for RAG retrieval.
        self._session_document_ids: set[str] = set()

    def add_document(self, file_path: str) -> str:
        """Ingest a file into this project (chunked + embedded for RAG) and
        pin it into the current session's payload."""
        if self.document_ingestor is None:
            raise RuntimeError("This Project was constructed without a DocumentIngestor")
        document_id = self.document_ingestor.add_document(self._record.id, file_path)
        self._session_document_ids.add(document_id)
        return document_id

    def _supports_caching(self) -> bool:
        return bool(self.provider_config) and getattr(
            self.provider_config, "SUPPORTS_PROMPT_CACHING", False
        )

    def _select_cached_documents(self) -> list:
        """The documents to send verbatim this turn (cache/decision.py):
        session-added documents always ride along on a caching provider (this
        is what performs the initial cache write); older documents only while
        their provider-side cache is still live. Everything else falls back to
        RAG retrieval. Non-caching providers never send documents verbatim."""
        if not self._supports_caching():
            return []
        selected = []
        for document in self.store.list_documents(self._record.id):
            if document.content is None:
                continue  # pre-content-column row: RAG only
            strategy = choose_strategy(
                provider_ttl_expires_at=document.provider_ttl_expires_at,
                supports_prompt_caching=True,
            )
            if document.id in self._session_document_ids or strategy is ContextStrategy.NATIVE_CACHE:
                selected.append(document)
        return selected

    def chat(self, messages: list[dict]) -> dict:
        current_message = messages[-1]["content"]

        cached_documents = self._select_cached_documents()
        retrieved_chunks = self.retriever.retrieve(
            self._record.id,
            current_message,
            exclude_document_ids=tuple(d.id for d in cached_documents),
        )

        payload = build_payload(
            retrieved_chunks,
            self._window.rolling_summary,
            self._window.verbatim_turns,
            current_message,
            cached_documents=[(d.filename, d.content) for d in cached_documents],
            cache_markers=bool(self.provider_config)
            and getattr(self.provider_config, "NEEDS_CACHE_MARKERS", False),
        )

        response = litellm.completion(model=self.chat_model, messages=payload, api_key=self.api_key)

        # The provider call succeeded, so verbatim documents are now in its
        # prompt cache: record the write (or the TTL-refreshing read — both
        # Anthropic semantics) so choose_strategy sees a live TTL next turn.
        if cached_documents:
            ttl = timedelta(seconds=getattr(self.provider_config, "PROMPT_CACHE_TTL_SECONDS", 0))
            expires_at = datetime.now(timezone.utc) + ttl
            for document in cached_documents:
                self.store.mark_document_used(document.id, cache_write=True, ttl_expires_at=expires_at)

        baseline_tokens = litellm.token_counter(
            model=self.chat_model,
            messages=self._raw_turns + [{"role": "user", "content": current_message}],
        )
        actual_tokens = litellm.token_counter(model=self.chat_model, messages=payload)
        self.tracker.log_call(actual_tokens, baseline_tokens)

        self._raw_turns.append({"role": "user", "content": current_message})
        self._window.add_turn({"role": "user", "content": current_message})

        assistant_content = response["choices"][0]["message"]["content"]
        self._raw_turns.append({"role": "assistant", "content": assistant_content})
        self._window.add_turn({"role": "assistant", "content": assistant_content})

        self.store.update_sub_conversation(self._sub_conversation.id, raw_turns=json.dumps(self._raw_turns))

        return response

    def end_session(self) -> None:
        summary = self._window.finalize()
        embedding = self.retriever.embedder.embed(summary)
        self.store.add_memory_chunk(self._record.id, self._sub_conversation.id, summary, embedding)
        self.store.update_sub_conversation(
            self._sub_conversation.id,
            summary=summary,
            ended_at=datetime.now(timezone.utc),
        )
        self.tracker.log_session_end()

        # Reset for the next sub-conversation in this project.
        self._sub_conversation = self.store.start_sub_conversation(self._record.id)
        self._window = SlidingWindow(self._summarizer, window_size=self._window_size)
        self._raw_turns = []
        self._session_document_ids = set()
