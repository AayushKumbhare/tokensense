"""Chunk and embed uploaded documents so cross-session reuse goes through RAG
retrieval instead of resending the full file verbatim (see project doc:
Static Content Strategy: Provider Caching vs. RAG)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .embedder import Embedder
from .store import Store

DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def hash_file(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DocumentIngestor:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def add_document(self, project_id: str, file_path: str) -> str:
        path = Path(file_path)
        raw_bytes = path.read_bytes()
        file_hash = hash_file(raw_bytes)

        existing = self.store.get_document_by_hash(project_id, file_hash)
        if existing is not None:
            return existing.id  # unchanged file already ingested — skip re-chunking/re-embedding

        text = raw_bytes.decode("utf-8", errors="ignore")
        chunks = chunk_text(text)
        embeddings = self.embedder.embed_many(chunks)

        document = self.store.add_document(
            project_id, filename=path.name, file_hash=file_hash, content=text
        )
        for content, embedding in zip(chunks, embeddings):
            self.store.add_document_chunk(project_id, document.id, content, embedding)
        return document.id
