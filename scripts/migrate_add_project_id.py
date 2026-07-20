"""One-time migration for installs created before the Phase 0 multi-project schema.

Adds the denormalized, required `project_id` column to `memory_chunks` and
`document_chunks` (backfilled through the existing joins), converts all child
foreign keys to ON DELETE CASCADE, and creates the supporting indexes.

Idempotent: safe to re-run; each step is skipped if already applied.

Usage:
    python scripts/migrate_add_project_id.py postgresql://user:pass@localhost:5432/tokensense
"""
from __future__ import annotations

import sys

from sqlalchemy import create_engine, text

STEPS: list[tuple[str, str]] = [
    (
        "add memory_chunks.project_id",
        "ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS project_id UUID",
    ),
    (
        "backfill memory_chunks.project_id",
        """
        UPDATE memory_chunks mc
        SET project_id = sc.project_id
        FROM sub_conversations sc
        WHERE mc.sub_conversation_id = sc.id AND mc.project_id IS NULL
        """,
    ),
    (
        "enforce memory_chunks.project_id NOT NULL",
        "ALTER TABLE memory_chunks ALTER COLUMN project_id SET NOT NULL",
    ),
    (
        "add document_chunks.project_id",
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS project_id UUID",
    ),
    (
        "backfill document_chunks.project_id",
        """
        UPDATE document_chunks dc
        SET project_id = d.project_id
        FROM documents d
        WHERE dc.document_id = d.id AND dc.project_id IS NULL
        """,
    ),
    (
        "enforce document_chunks.project_id NOT NULL",
        "ALTER TABLE document_chunks ALTER COLUMN project_id SET NOT NULL",
    ),
    # Recreate every child FK with ON DELETE CASCADE so deleting a project row
    # removes the whole tree.
    (
        "cascade sub_conversations -> projects",
        """
        ALTER TABLE sub_conversations
            DROP CONSTRAINT IF EXISTS sub_conversations_project_id_fkey,
            ADD CONSTRAINT sub_conversations_project_id_fkey
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        """,
    ),
    (
        "cascade memory_chunks -> sub_conversations",
        """
        ALTER TABLE memory_chunks
            DROP CONSTRAINT IF EXISTS memory_chunks_sub_conversation_id_fkey,
            ADD CONSTRAINT memory_chunks_sub_conversation_id_fkey
                FOREIGN KEY (sub_conversation_id) REFERENCES sub_conversations(id) ON DELETE CASCADE
        """,
    ),
    (
        "cascade memory_chunks -> projects",
        """
        ALTER TABLE memory_chunks
            DROP CONSTRAINT IF EXISTS memory_chunks_project_id_fkey,
            ADD CONSTRAINT memory_chunks_project_id_fkey
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        """,
    ),
    (
        "cascade documents -> projects",
        """
        ALTER TABLE documents
            DROP CONSTRAINT IF EXISTS documents_project_id_fkey,
            ADD CONSTRAINT documents_project_id_fkey
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        """,
    ),
    (
        "cascade document_chunks -> documents",
        """
        ALTER TABLE document_chunks
            DROP CONSTRAINT IF EXISTS document_chunks_document_id_fkey,
            ADD CONSTRAINT document_chunks_document_id_fkey
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        """,
    ),
    (
        "cascade document_chunks -> projects",
        """
        ALTER TABLE document_chunks
            DROP CONSTRAINT IF EXISTS document_chunks_project_id_fkey,
            ADD CONSTRAINT document_chunks_project_id_fkey
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        """,
    ),
    (
        "index memory_chunks.project_id",
        "CREATE INDEX IF NOT EXISTS ix_memory_chunks_project_id ON memory_chunks (project_id)",
    ),
    (
        "index document_chunks.project_id",
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_project_id ON document_chunks (project_id)",
    ),
    (
        "index sub_conversations.project_id",
        "CREATE INDEX IF NOT EXISTS ix_sub_conversations_project_id ON sub_conversations (project_id)",
    ),
    (
        "index documents.project_id",
        "CREATE INDEX IF NOT EXISTS ix_documents_project_id ON documents (project_id)",
    ),
    (
        "HNSW index on memory_chunks.embedding",
        """
        CREATE INDEX IF NOT EXISTS ix_memory_chunks_embedding_hnsw
        ON memory_chunks USING hnsw (embedding vector_cosine_ops)
        """,
    ),
    (
        "HNSW index on document_chunks.embedding",
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
        ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """,
    ),
]


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)

    engine = create_engine(sys.argv[1])
    with engine.begin() as conn:
        for label, sql in STEPS:
            print(f"-> {label}")
            conn.execute(text(sql))
    print("Migration complete.")


if __name__ == "__main__":
    main()
