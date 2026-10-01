from pathlib import Path

from app.rag.discovery.document_registry import (
    DocumentRegistry,
)


def test_register_document(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "test document",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    record = registry.register(
        document,
    )

    assert record.identity.path == document
    assert record.identity.filename == "report.pdf"
    assert record.identity.extension == ".pdf"
    assert record.indexed is False

    assert registry.contains(document)
    assert registry.count() == 1


def test_document_is_not_indexed_initially(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "test document",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    registry.register(document)

    assert registry.is_indexed(document) is False
    assert registry.needs_reindex(document) is True


def test_mark_document_indexed(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "test document",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    registry.register(document)

    registry.mark_indexed(document)

    assert registry.is_indexed(document) is True
    assert registry.needs_reindex(document) is False


def test_modified_document_requires_reindex(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "original content",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    registry.register(document)

    registry.mark_indexed(document)

    assert registry.is_indexed(document) is True

    document.write_text(
        "modified content that is different",
        encoding="utf-8",
    )

    assert registry.needs_reindex(document) is True
    assert registry.is_indexed(document) is False


def test_removed_document_requires_reindex(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "test document",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    registry.register(document)

    registry.mark_indexed(document)

    assert registry.is_indexed(document) is True

    document.unlink()

    assert registry.needs_reindex(document) is True
    assert registry.is_indexed(document) is False


def test_mark_not_indexed(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "test document",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    registry.register(
        document,
        indexed=True,
    )

    assert registry.is_indexed(document) is True

    registry.mark_not_indexed(document)

    assert registry.is_indexed(document) is False
    assert registry.needs_reindex(document) is True


def test_remove_document(tmp_path):

    document = tmp_path / "report.pdf"

    document.write_text(
        "test document",
        encoding="utf-8",
    )

    registry = DocumentRegistry()

    registry.register(document)

    assert registry.contains(document)

    removed = registry.remove(document)

    assert removed is True
    assert registry.contains(document) is False
    assert registry.count() == 0