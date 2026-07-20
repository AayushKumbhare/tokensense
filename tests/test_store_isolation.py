"""Phase 0 guards: every retrieval statement is project-scoped, and the schema
enforces isolation (denormalized project_id, NOT NULL, ON DELETE CASCADE).

These compile the actual statements/DDL rather than hitting a live database,
so the guarantee is checked structurally on every test run.
"""
from sqlalchemy.dialects import postgresql

from tokensense.memory import store as store_module
from tokensense.memory.store import (
    EMBEDDING_DIM,
    DocumentChunkRecord,
    MemoryChunkRecord,
    document_chunk_topk_stmt,
    memory_chunk_topk_stmt,
    memory_chunk_topk_with_sources_stmt,
)

EMBEDDING = [0.0] * EMBEDDING_DIM


def _compiled(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_memory_retrieval_is_project_scoped():
    sql = _compiled(memory_chunk_topk_stmt("pid", EMBEDDING, 5))
    assert "memory_chunks.project_id =" in sql


def test_memory_retrieval_with_sources_is_project_scoped():
    sql = _compiled(memory_chunk_topk_with_sources_stmt("pid", EMBEDDING, 5))
    assert "memory_chunks.project_id =" in sql


def test_document_retrieval_is_project_scoped():
    sql = _compiled(document_chunk_topk_stmt("pid", EMBEDDING, 3))
    assert "document_chunks.project_id =" in sql


def test_document_retrieval_stays_project_scoped_with_exclusions():
    sql = _compiled(document_chunk_topk_stmt("pid", EMBEDDING, 3, exclude_document_ids=("d1",)))
    assert "document_chunks.project_id =" in sql
    assert "NOT IN" in sql.upper()


def test_retrieval_builders_require_project_id():
    import inspect

    for builder in (memory_chunk_topk_stmt, memory_chunk_topk_with_sources_stmt, document_chunk_topk_stmt):
        params = list(inspect.signature(builder).parameters)
        assert params[0] == "project_id"


def test_no_cross_project_retrieval_exists():
    assert not hasattr(store_module.Store, "retrieve_cross_project")


def test_chunk_tables_have_required_indexed_project_id():
    for record in (MemoryChunkRecord, DocumentChunkRecord):
        col = record.__table__.c.project_id
        assert col.nullable is False
        assert col.index is True


def test_all_child_fks_cascade_on_delete():
    from tokensense.memory.store import Base

    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            assert fk.ondelete == "CASCADE", f"{table.name} FK to {fk.column.table.name} must cascade"
