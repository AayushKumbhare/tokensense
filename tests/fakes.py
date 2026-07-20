"""In-memory stand-ins for the DB/network-backed pieces, so transport and
engine logic is testable without Postgres, Ollama, or provider APIs."""
from __future__ import annotations

import itertools
from types import SimpleNamespace

from tokensense.summarizers.base import BaseSummarizer

_ids = itertools.count(1)


class FakeSummarizer(BaseSummarizer):
    def __init__(self):
        self.calls: list[str] = []

    def _complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Echo the prompt so stored summaries carry the original turn text,
        # letting tests assert on real content crossing (or not crossing)
        # project boundaries.
        return f"summary#{len(self.calls)}: {prompt}"


class FakeEmbedder:
    model = "fake-embedding"

    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FakeStore:
    """Mirrors the Store interface, enforcing the same project scoping."""

    def __init__(self):
        self.projects: dict[str, SimpleNamespace] = {}
        self.sub_conversations: dict[str, SimpleNamespace] = {}
        self.memory_chunks: list[SimpleNamespace] = []
        self.documents: list[SimpleNamespace] = []
        self.document_chunks: list[SimpleNamespace] = []

    def get_or_create_project(self, name: str) -> SimpleNamespace:
        if name not in self.projects:
            self.projects[name] = SimpleNamespace(id=f"proj-{name}", name=name)
        return self.projects[name]

    def start_sub_conversation(self, project_id: str, external_id=None) -> SimpleNamespace:
        sub = SimpleNamespace(
            id=f"sub-{next(_ids)}",
            project_id=project_id,
            external_id=external_id,
            raw_turns="[]",
            summary=None,
            ended_at=None,
        )
        self.sub_conversations[sub.id] = sub
        return sub

    def get_sub_conversation_by_external_id(self, project_id, external_id):
        for sub in self.sub_conversations.values():
            if sub.project_id == project_id and sub.external_id == external_id:
                return sub
        return None

    def replace_memory_chunks_for_sub_conversation(self, project_id, sub_conversation_id, content, embedding):
        self.memory_chunks = [c for c in self.memory_chunks if c.sub_conversation_id != sub_conversation_id]
        return self.add_memory_chunk(project_id, sub_conversation_id, content, embedding)

    def update_sub_conversation(self, sub_conversation_id, *, raw_turns=None, summary=None, ended_at=None):
        sub = self.sub_conversations[sub_conversation_id]
        if raw_turns is not None:
            sub.raw_turns = raw_turns
        if summary is not None:
            sub.summary = summary
        if ended_at is not None:
            sub.ended_at = ended_at

    def add_memory_chunk(self, project_id, sub_conversation_id, content, embedding):
        chunk = SimpleNamespace(
            id=f"mem-{next(_ids)}",
            project_id=project_id,
            sub_conversation_id=sub_conversation_id,
            content=content,
            embedding=embedding,
        )
        self.memory_chunks.append(chunk)
        return chunk

    def top_k_memory_chunks(self, project_id, query_embedding, k=5):
        return [c.content for c in self.memory_chunks if c.project_id == project_id][:k]

    def top_k_memory_chunks_with_sources(self, project_id, query_embedding, k=5):
        return [
            (c.content, self.sub_conversations[c.sub_conversation_id].raw_turns)
            for c in self.memory_chunks
            if c.project_id == project_id
        ][:k]

    def get_document_by_hash(self, project_id, file_hash):
        for doc in self.documents:
            if doc.project_id == project_id and doc.file_hash == file_hash:
                return doc
        return None

    def add_document(self, project_id, filename, file_hash, content=None):
        doc = SimpleNamespace(
            id=f"doc-{next(_ids)}",
            project_id=project_id,
            filename=filename,
            file_hash=file_hash,
            content=content,
            added_at=None,
            last_used_at=None,
            last_cache_write_at=None,
            provider_ttl_expires_at=None,
        )
        self.documents.append(doc)
        return doc

    def list_documents(self, project_id):
        return [d for d in self.documents if d.project_id == project_id]

    def add_document_chunk(self, project_id, document_id, content, embedding):
        self.document_chunks.append(
            SimpleNamespace(project_id=project_id, document_id=document_id, content=content, embedding=embedding)
        )

    def top_k_document_chunks(self, project_id, query_embedding, k=3, exclude_document_ids=()):
        return [
            (c.document_id, c.content)
            for c in self.document_chunks
            if c.project_id == project_id and c.document_id not in exclude_document_ids
        ][:k]

    def mark_document_used(self, document_id, *, cache_write=False, ttl_expires_at=None):
        doc = next(d for d in self.documents if d.id == document_id)
        doc.last_used_at = "used"
        if cache_write:
            doc.last_cache_write_at = "written"
            doc.provider_ttl_expires_at = ttl_expires_at


def make_engine(monkeypatch=None, **config_overrides):
    """A ServerEngine wired to fakes. Token counting is patched to a simple
    character count when a monkeypatch fixture is supplied."""
    from tokensense.memory.retriever import Retriever
    from tokensense.server.config import ServerConfig
    from tokensense.server.engine import ServerEngine

    config = ServerConfig(db_url="unused", **config_overrides)
    engine = ServerEngine(config, store=FakeStore())
    engine.embedder = FakeEmbedder()
    engine.retriever = Retriever(engine.store, engine.embedder, top_k=config.top_k)
    engine.summarizer = FakeSummarizer()

    if monkeypatch is not None:
        import litellm

        monkeypatch.setattr(
            litellm,
            "token_counter",
            lambda model=None, messages=None: sum(len(str(m.get("content", ""))) for m in messages),
        )
    return engine
