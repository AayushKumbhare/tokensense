"""pgvector-backed persistence for projects, sub-conversations, memory chunks, and documents.

Project isolation (see revised architecture plan, Phase 0): `project_id` is a
required, indexed, denormalized column on both chunk tables — not merely
reachable via join — and every retrieval statement is built by a module-level
builder that takes `project_id` as a required argument, so no code path can
omit the filter. Cross-project retrieval does not exist; if it is ever wanted
it must be added as an explicitly named `retrieve_cross_project(...)`, never
as a default.

Index strategy: a btree index on `project_id` plus an HNSW index on the
embedding column (pgvector HNSW indexes are single-column, so a true composite
is not possible). Partitioning by `PARTITION BY LIST (project_id)` is
deliberately deferred until the benchmark harness shows single-index
degradation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text, create_engine, select, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

# Must match the output dimension of whatever embedding model is configured.
# 768 is nomic-embed-text's native dimension (the local-first default); OpenAI
# text-embedding-3-* models are requested at 768 via their `dimensions`
# parameter so both fit the same schema. Changing to a model with a different
# dimension requires a fresh schema and re-embedding all stored chunks —
# Store raises at startup if the existing tables disagree (see decisions.md).
EMBEDDING_DIM = 768


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProjectRecord(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)

    sub_conversations = relationship(
        "SubConversationRecord", back_populates="project", cascade="all, delete-orphan"
    )
    documents = relationship("DocumentRecord", back_populates="project", cascade="all, delete-orphan")


class SubConversationRecord(Base):
    __tablename__ = "sub_conversations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Host-tool session identity (e.g. a Claude Code session id) so re-ingesting
    # the same session — SessionEnd can fire more than once, and resumed
    # sessions re-emit a grown transcript — updates in place instead of
    # duplicating memory.
    external_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    raw_turns = Column(Text, nullable=False, default="[]")  # JSON-encoded list of {role, content}
    summary = Column(Text, nullable=True)

    project = relationship("ProjectRecord", back_populates="sub_conversations")
    memory_chunks = relationship(
        "MemoryChunkRecord", back_populates="sub_conversation", cascade="all, delete-orphan"
    )


class MemoryChunkRecord(Base):
    __tablename__ = "memory_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    # Denormalized from sub_conversations so retrieval filters on this table
    # directly instead of trusting a join for isolation.
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sub_conversation_id = Column(
        UUID(as_uuid=False), ForeignKey("sub_conversations.id", ondelete="CASCADE"), nullable=False
    )
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)

    sub_conversation = relationship("SubConversationRecord", back_populates="memory_chunks")

    __table_args__ = (
        Index(
            "ix_memory_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    # Full original text, kept so the NATIVE_CACHE strategy can resend the
    # exact bytes the provider cached (cache hits require identical prefixes).
    # Nullable for rows ingested before this column existed — those can only
    # be served via RAG chunks.
    content = Column(Text, nullable=True)
    added_at = Column(DateTime(timezone=True), default=_now)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_cache_write_at = Column(DateTime(timezone=True), nullable=True)
    provider_ttl_expires_at = Column(DateTime(timezone=True), nullable=True)

    project = relationship("ProjectRecord", back_populates="documents")
    chunks = relationship("DocumentChunkRecord", back_populates="document", cascade="all, delete-orphan")


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    # Denormalized from documents, same rationale as memory_chunks.project_id.
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id = Column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    document = relationship("DocumentRecord", back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# -- retrieval statement builders ------------------------------------------------------
# Every retrieval statement requires project_id; there is intentionally no
# builder without it.


def memory_chunk_topk_stmt(project_id: str, query_embedding: list[float], k: int):
    return (
        select(MemoryChunkRecord.content)
        .where(MemoryChunkRecord.project_id == project_id)
        .order_by(MemoryChunkRecord.embedding.cosine_distance(query_embedding))
        .limit(k)
    )


def memory_chunk_topk_with_sources_stmt(project_id: str, query_embedding: list[float], k: int):
    """Returns (content, raw_turns) rows — each retrieved summary alongside the
    raw session it condenses — so the MCP transport can account for the tokens
    the summary replaces (decisions.md #7). The join is for the raw_turns
    payload only; scoping still filters on the chunk table's own project_id."""
    return (
        select(MemoryChunkRecord.content, SubConversationRecord.raw_turns)
        .join(
            SubConversationRecord,
            MemoryChunkRecord.sub_conversation_id == SubConversationRecord.id,
        )
        .where(MemoryChunkRecord.project_id == project_id)
        .order_by(MemoryChunkRecord.embedding.cosine_distance(query_embedding))
        .limit(k)
    )


def document_chunk_topk_stmt(
    project_id: str,
    query_embedding: list[float],
    k: int,
    exclude_document_ids: tuple[str, ...] = (),
):
    """Returns (document_id, content) rows so callers can mark usage on the
    parent document. `exclude_document_ids` skips documents already included
    in the payload verbatim via the NATIVE_CACHE strategy."""
    stmt = select(DocumentChunkRecord.document_id, DocumentChunkRecord.content).where(
        DocumentChunkRecord.project_id == project_id
    )
    if exclude_document_ids:
        stmt = stmt.where(DocumentChunkRecord.document_id.notin_(exclude_document_ids))
    return stmt.order_by(DocumentChunkRecord.embedding.cosine_distance(query_embedding)).limit(k)


class Store:
    """High-level read/write operations. Owns the engine and session factory so
    callers never touch SQLAlchemy sessions directly."""

    def __init__(self, db_url: str, ensure_schema: bool = True):
        """`ensure_schema=False` skips CREATE EXTENSION / CREATE TABLE — for
        restricted roles (see scripts/create_restricted_role.py) that only
        have DML grants on already-provisioned tables and would get a
        permission error attempting DDL, even idempotent DDL against objects
        that already exist (Postgres checks CREATE privilege before checking
        IF NOT EXISTS)."""
        self.engine = create_engine(db_url)
        if ensure_schema:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
        with self.engine.connect() as conn:
            existing_dim = conn.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute"
                    " WHERE attrelid = to_regclass('memory_chunks') AND attname = 'embedding'"
                )
            ).scalar()
        if existing_dim is not None and existing_dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Existing schema stores {existing_dim}-dim embeddings but this build expects "
                f"{EMBEDDING_DIM}. Embeddings from different models are not comparable, so there is "
                "no in-place migration: back up what you need, drop the tables (projects, "
                "sub_conversations, memory_chunks, documents, document_chunks), and re-embed."
            )
        if ensure_schema:
            Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # -- projects / sub-conversations -------------------------------------------------

    def get_or_create_project(self, name: str) -> ProjectRecord:
        with self.Session() as session:
            project = session.query(ProjectRecord).filter_by(name=name).one_or_none()
            if project is None:
                project = ProjectRecord(name=name)
                session.add(project)
                session.commit()
                session.refresh(project)
            return project

    def start_sub_conversation(self, project_id: str, external_id: str | None = None) -> SubConversationRecord:
        with self.Session() as session:
            sub = SubConversationRecord(project_id=project_id, raw_turns="[]", external_id=external_id)
            session.add(sub)
            session.commit()
            session.refresh(sub)
            return sub

    def get_sub_conversation_by_external_id(
        self, project_id: str, external_id: str
    ) -> SubConversationRecord | None:
        with self.Session() as session:
            return (
                session.query(SubConversationRecord)
                .filter_by(project_id=project_id, external_id=external_id)
                .one_or_none()
            )

    def replace_memory_chunks_for_sub_conversation(
        self, project_id: str, sub_conversation_id: str, content: str, embedding: list[float]
    ) -> MemoryChunkRecord:
        """Idempotent re-ingestion: drop the sub-conversation's previous memory
        and store the fresh summary as its single chunk."""
        with self.Session() as session:
            session.query(MemoryChunkRecord).filter_by(
                sub_conversation_id=sub_conversation_id
            ).delete()
            chunk = MemoryChunkRecord(
                project_id=project_id,
                sub_conversation_id=sub_conversation_id,
                content=content,
                embedding=embedding,
            )
            session.add(chunk)
            session.commit()
            session.refresh(chunk)
            return chunk

    def update_sub_conversation(
        self,
        sub_conversation_id: str,
        *,
        raw_turns: str | None = None,
        summary: str | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        with self.Session() as session:
            sub = session.get(SubConversationRecord, sub_conversation_id)
            if raw_turns is not None:
                sub.raw_turns = raw_turns
            if summary is not None:
                sub.summary = summary
            if ended_at is not None:
                sub.ended_at = ended_at
            session.commit()

    # -- memory chunks ------------------------------------------------------------------

    def add_memory_chunk(
        self, project_id: str, sub_conversation_id: str, content: str, embedding: list[float]
    ) -> MemoryChunkRecord:
        with self.Session() as session:
            chunk = MemoryChunkRecord(
                project_id=project_id,
                sub_conversation_id=sub_conversation_id,
                content=content,
                embedding=embedding,
            )
            session.add(chunk)
            session.commit()
            session.refresh(chunk)
            return chunk

    def top_k_memory_chunks(self, project_id: str, query_embedding: list[float], k: int = 5) -> list[str]:
        with self.Session() as session:
            return list(session.scalars(memory_chunk_topk_stmt(project_id, query_embedding, k)))

    def top_k_memory_chunks_with_sources(
        self, project_id: str, query_embedding: list[float], k: int = 5
    ) -> list[tuple[str, str]]:
        """Returns (content, raw_turns) pairs, nearest first."""
        with self.Session() as session:
            return [
                (row.content, row.raw_turns)
                for row in session.execute(
                    memory_chunk_topk_with_sources_stmt(project_id, query_embedding, k)
                )
            ]

    # -- documents ------------------------------------------------------------------------

    def get_document_by_hash(self, project_id: str, file_hash: str) -> DocumentRecord | None:
        with self.Session() as session:
            return (
                session.query(DocumentRecord)
                .filter_by(project_id=project_id, file_hash=file_hash)
                .one_or_none()
            )

    def add_document(
        self, project_id: str, filename: str, file_hash: str, content: str | None = None
    ) -> DocumentRecord:
        with self.Session() as session:
            document = DocumentRecord(
                project_id=project_id, filename=filename, file_hash=file_hash, content=content
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document

    def list_documents(self, project_id: str) -> list[DocumentRecord]:
        with self.Session() as session:
            return list(
                session.query(DocumentRecord)
                .filter_by(project_id=project_id)
                .order_by(DocumentRecord.added_at, DocumentRecord.id)
            )

    def add_document_chunk(
        self, project_id: str, document_id: str, content: str, embedding: list[float]
    ) -> None:
        with self.Session() as session:
            chunk = DocumentChunkRecord(
                project_id=project_id, document_id=document_id, content=content, embedding=embedding
            )
            session.add(chunk)
            session.commit()

    def top_k_document_chunks(
        self,
        project_id: str,
        query_embedding: list[float],
        k: int = 3,
        exclude_document_ids: tuple[str, ...] = (),
    ) -> list[tuple[str, str]]:
        """Returns (document_id, content) pairs, nearest first."""
        with self.Session() as session:
            return [
                (row.document_id, row.content)
                for row in session.execute(
                    document_chunk_topk_stmt(project_id, query_embedding, k, exclude_document_ids)
                )
            ]

    def mark_document_used(
        self, document_id: str, *, cache_write: bool = False, ttl_expires_at: datetime | None = None
    ) -> None:
        with self.Session() as session:
            document = session.get(DocumentRecord, document_id)
            document.last_used_at = _now()
            if cache_write:
                document.last_cache_write_at = _now()
                document.provider_ttl_expires_at = ttl_expires_at
            session.commit()
