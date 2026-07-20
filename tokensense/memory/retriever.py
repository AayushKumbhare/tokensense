"""RAG retrieval over a project's memory chunks and document chunks."""
from __future__ import annotations

from .embedder import Embedder
from .store import Store


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self, project_id: str, query: str, exclude_document_ids: tuple[str, ...] = ()
    ) -> list[str]:
        """Top-K memory and document chunks for the query. Documents whose
        full text is already in the payload via the NATIVE_CACHE strategy are
        excluded so their content isn't sent twice. Documents that contribute
        a retrieved chunk get their `last_used_at` stamped."""
        chunks, _ = self.retrieve_with_sources(project_id, query, exclude_document_ids)
        return chunks

    def retrieve_with_sources(
        self, project_id: str, query: str, exclude_document_ids: tuple[str, ...] = ()
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Like retrieve, but also returns (content, raw_turns) for each memory
        hit so callers can account for the raw session tokens the summary
        replaces (the MCP savings model, decisions.md #7). Document chunks
        appear only in the combined chunk list — they replace no session."""
        query_embedding = self.embedder.embed(query)
        memory_rows = self.store.top_k_memory_chunks_with_sources(
            project_id, query_embedding, k=self.top_k
        )
        document_rows = self.store.top_k_document_chunks(
            project_id, query_embedding, k=self.top_k, exclude_document_ids=exclude_document_ids
        )
        for document_id in dict.fromkeys(doc_id for doc_id, _ in document_rows):
            self.store.mark_document_used(document_id)
        chunks = [content for content, _ in memory_rows] + [content for _, content in document_rows]
        return chunks, memory_rows
