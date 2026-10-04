from pathlib import Path
from types import SimpleNamespace

from app.rag.indexing.document_indexer import DocumentIndexer
from app.rag.access.document_access_manager import DocumentAccessManager


class FakeRegistry:
    def __init__(self, record, indexed=True):
        self.record = record
        self.indexed = indexed

    def is_indexed(self, path):
        return self.indexed

    def get(self, value):
        if str(value) == self.record.document_id:
            return self.record
        try:
            if str(Path(value).resolve()).lower() == str(Path(self.record.source_path).resolve()).lower():
                return self.record
        except (TypeError, ValueError):
            pass
        return None

    def find_by_path(self, path):
        return self.record

    def find_by_name(self, name):
        return self.record if name == self.record.file_name else None

    def list_documents(self):
        return [self.record]


class FakeBackend:
    def __init__(self, count):
        self.count = count

    def document_chunk_count(self, document_id):
        return self.count


def test_registry_record_is_not_enough_when_qdrant_or_bm25_is_missing(tmp_path):
    path = tmp_path / "phoenix_evaluation.docx"
    path.write_text("x", encoding="utf-8")
    record = SimpleNamespace(
        document_id="doc-1",
        file_name=path.name,
        source_path=str(path),
        chunk_count=3,
        status="indexed",
    )

    indexer = DocumentIndexer.__new__(DocumentIndexer)
    indexer.registry = FakeRegistry(record, indexed=True)
    indexer.qdrant_store = FakeBackend(0)
    indexer.bm25_store = FakeBackend(3)

    assert indexer.is_document_indexed(path) is False


def test_registry_record_is_healthy_when_all_backends_match(tmp_path):
    path = tmp_path / "phoenix_evaluation.docx"
    path.write_text("x", encoding="utf-8")
    record = SimpleNamespace(
        document_id="doc-1",
        file_name=path.name,
        source_path=str(path),
        chunk_count=3,
        status="indexed",
    )

    indexer = DocumentIndexer.__new__(DocumentIndexer)
    indexer.registry = FakeRegistry(record, indexed=True)
    indexer.qdrant_store = FakeBackend(3)
    indexer.bm25_store = FakeBackend(3)

    assert indexer.is_document_indexed(path) is True


def test_extension_is_never_resolved_by_shared_stem(tmp_path):
    docx = tmp_path / "phoenix_evaluation.docx"
    pdf = tmp_path / "phoenix_evaluation.pdf"
    docx.write_text("docx", encoding="utf-8")
    pdf.write_text("pdf", encoding="utf-8")

    docx_record = SimpleNamespace(
        document_id="docx-1",
        file_name=docx.name,
        source_path=str(docx),
        chunk_count=1,
        status="indexed",
    )

    class Registry(FakeRegistry):
        def find_by_name(self, name):
            return docx_record if name.lower() == docx.name.lower() else None

        def find_by_path(self, path):
            return docx_record if str(Path(path).resolve()).lower() == str(docx.resolve()).lower() else None

        def list_documents(self):
            return [docx_record]

    class Indexer:
        def is_document_indexed(self, path):
            return False

    manager = DocumentAccessManager(
        document_registry=Registry(docx_record),
        document_discovery=SimpleNamespace(search=lambda *args, **kwargs: []),
        document_indexer=Indexer(),
        default_search_roots=[str(tmp_path)],
    )

    assert manager._resolve_registered(pdf.name) is None
