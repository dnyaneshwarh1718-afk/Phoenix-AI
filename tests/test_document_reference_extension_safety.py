from pathlib import Path

from app.rag.access.document_access_manager import DocumentAccessManager
from app.rag.registry.document_registry import DocumentRegistry


class _NoOpIndexer:
    def index_document(self, path):
        raise AssertionError("The test should resolve the exact file before indexing.")


def _manager(tmp_path):
    class _Discovery:
        def search(self, root, reference, limit):
            return []

    return DocumentAccessManager(
        document_registry=DocumentRegistry(str(tmp_path / "registry.json")),
        document_discovery=_Discovery(),
        document_indexer=_NoOpIndexer(),
        default_search_roots=[str(tmp_path)],
    )


def test_extension_prevents_same_stem_cross_document_resolution(tmp_path):
    """A .pdf reference must never resolve to a registered .docx with the same stem."""
    docx = tmp_path / "phoenix_evaluation.docx"
    pdf = tmp_path / "phoenix_evaluation.pdf"
    docx.write_text("docx", encoding="utf-8")
    pdf.write_text("pdf", encoding="utf-8")

    manager = _manager(tmp_path)
    manager.document_registry.register(
        document_id="docx-id",
        file_name=docx.name,
        source_path=str(docx),
        file_type=".docx",
        chunk_count=1,
    )

    result = manager.prepare_document(str(pdf), auto_index=False)

    assert result.matched_path == str(pdf.resolve())
    assert result.document is None
    assert result.status == "discovered_not_indexed"


def test_extensionless_reference_can_use_unique_stem(tmp_path):
    docx = tmp_path / "unique_document.docx"
    docx.write_text("docx", encoding="utf-8")

    manager = _manager(tmp_path)
    record = manager.document_registry.register(
        document_id="docx-id",
        file_name=docx.name,
        source_path=str(docx),
        file_type=".docx",
        chunk_count=1,
    )

    result = manager.prepare_document("unique_document", auto_index=False)

    assert result.document is record
    assert result.status == "discovered_not_indexed" or result.ready
