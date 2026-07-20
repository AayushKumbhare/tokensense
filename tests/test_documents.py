from tokensense.memory.documents import DocumentIngestor, chunk_text, hash_file

from .fakes import FakeEmbedder, FakeStore


def test_chunk_text_splits_long_text():
    text = "x" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("short") == ["short"]


def test_hash_file_is_deterministic():
    assert hash_file(b"hello") == hash_file(b"hello")
    assert hash_file(b"hello") != hash_file(b"world")


def test_shared_file_duplicated_per_project(tmp_path):
    """Phase 5 decision (docs/decisions.md): a file shared across projects is
    stored once per project — file_hash dedup is scoped within a project and
    never lets one project's chunks be reachable from another."""
    spec = tmp_path / "shared-api-spec.md"
    spec.write_text("GET /v1/things returns a list of things")

    store = FakeStore()
    ingestor = DocumentIngestor(store, FakeEmbedder())

    doc_a = ingestor.add_document("proj-a", str(spec))
    doc_b = ingestor.add_document("proj-b", str(spec))

    # Two independent rows for the same physical file.
    assert doc_a != doc_b
    assert len(store.documents) == 2
    assert {d.project_id for d in store.documents} == {"proj-a", "proj-b"}
    # Chunks are scoped to their own project.
    assert store.top_k_document_chunks("proj-a", [0.0]) == [
        (doc_a, "GET /v1/things returns a list of things")
    ]
    assert all(c.project_id in ("proj-a", "proj-b") for c in store.document_chunks)

    # Within one project, re-ingesting the unchanged file is a no-op.
    assert ingestor.add_document("proj-a", str(spec)) == doc_a
    assert len(store.documents) == 2
