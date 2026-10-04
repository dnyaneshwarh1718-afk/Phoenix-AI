from pathlib import Path

from app.rag.access.document_access_manager import DocumentAccessManager
from app.rag.registry.document_registry import DocumentRegistry


class FakeBackendIndexer:
    def __init__(self, qdrant_count=0, bm25_count=0):
        self.qdrant_count = qdrant_count
        self.bm25_count = bm25_count
        self.calls = []

    def is_index_consistent(self, path, document_id=None):
        return self.qdrant_count > 0 and self.bm25_count > 0

    def index_document(self, path, force_reindex=False):
        self.calls.append((str(path), force_reindex))
        class Status:
            name = "REINDEXED" if force_reindex else "INDEXED"
        class Result:
            status = Status()
            document_id = "doc-1"
            message = "reindexed"
        return Result()


def test_registry_does_not_mask_missing_backend_index(tmp_path):
    path = tmp_path / "phoenix.txt"
    path.write_text("Phoenix uses Qdrant.", encoding="utf-8")
    registry = DocumentRegistry(str(tmp_path / "registry.json"))
    registry.register("doc-1", path.name, str(path), ".txt", 1)

    indexer = FakeBackendIndexer(qdrant_count=0, bm25_count=0)
    manager = DocumentAccessManager(
        document_registry=registry,
        document_discovery=object(),
        document_indexer=indexer,
        default_search_roots=[str(tmp_path)],
    )

    result = manager.prepare_document(str(path), auto_index=True)

    assert result.status == "indexed"
    assert indexer.calls == [(str(path), True)]


def test_consistent_backends_are_reused(tmp_path):
    path = tmp_path / "phoenix.txt"
    path.write_text("Phoenix uses Qdrant.", encoding="utf-8")
    registry = DocumentRegistry(str(tmp_path / "registry.json"))
    registry.register("doc-1", path.name, str(path), ".txt", 1)
    indexer = FakeBackendIndexer(qdrant_count=1, bm25_count=1)
    manager = DocumentAccessManager(
        document_registry=registry,
        document_discovery=object(),
        document_indexer=indexer,
        default_search_roots=[str(tmp_path)],
    )

    result = manager.prepare_document(str(path), auto_index=True)

    assert result.ready
    assert result.status == "ready"
    assert indexer.calls == []
