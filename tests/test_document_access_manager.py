from app.rag.discovery.document_access_manager import (
    AccessAction,
    AccessStatus,
    DocumentAccessManager,
)

from app.rag.discovery.document_registry import (
    DocumentRegistry,
)

from app.rag.discovery.local_drive_finder import (
    LocalDriveDocumentFinder,
)


def create_manager(tmp_path):
    finder = LocalDriveDocumentFinder(
        primary_root=tmp_path,
        downloads_root=tmp_path / "downloads",
    )

    registry = DocumentRegistry()

    manager = DocumentAccessManager(
        finder=finder,
        registry=registry,
    )

    return manager, registry


def test_document_found(tmp_path):

    document = tmp_path / "SQL INTERVIEW QUESTIONS.docx"

    document.write_text(
        "SQL interview questions",
        encoding="utf-8",
    )

    manager, registry = create_manager(tmp_path)

    result = manager.locate(
        "SQL INTERVIEW QUESTIONS.docx"
    )

    assert result.status == AccessStatus.FOUND

    assert result.action == AccessAction.INDEX_DOCUMENT

    assert result.path == document

    assert result.filename == (
        "SQL INTERVIEW QUESTIONS.docx"
    )

    assert registry.is_indexed(document) is False


def test_document_found_by_stem(tmp_path):

    document = tmp_path / "SQL INTERVIEW QUESTIONS.docx"

    document.write_text(
        "SQL interview questions",
        encoding="utf-8",
    )

    manager, _ = create_manager(tmp_path)

    result = manager.locate(
        "SQL INTERVIEW QUESTIONS"
    )

    assert result.status == AccessStatus.FOUND

    assert result.action == AccessAction.INDEX_DOCUMENT

    assert result.path == document


def test_already_indexed_document_uses_existing_index(
    tmp_path,
):

    document = tmp_path / "SQL INTERVIEW QUESTIONS.docx"

    document.write_text(
        "SQL interview questions",
        encoding="utf-8",
    )

    manager, registry = create_manager(tmp_path)

    registry.register(document)

    registry.mark_indexed(document)

    result = manager.locate(
        "SQL INTERVIEW QUESTIONS.docx"
    )

    assert result.status == AccessStatus.FOUND

    assert (
        result.action
        == AccessAction.USE_EXISTING_INDEX
    )

    assert result.path == document


def test_modified_document_requires_reindex(
    tmp_path,
):

    document = tmp_path / "SQL INTERVIEW QUESTIONS.docx"

    document.write_text(
        "original content",
        encoding="utf-8",
    )

    manager, registry = create_manager(tmp_path)

    registry.register(document)

    registry.mark_indexed(document)

    assert registry.is_indexed(document) is True

    document.write_text(
        "modified content that is different",
        encoding="utf-8",
    )

    result = manager.locate(
        "SQL INTERVIEW QUESTIONS.docx"
    )

    assert result.status == AccessStatus.FOUND

    assert result.action == AccessAction.INDEX_DOCUMENT

    assert result.path == document


def test_document_not_found(tmp_path):

    manager, _ = create_manager(tmp_path)

    result = manager.locate(
        "this_document_does_not_exist"
    )

    assert result.status == AccessStatus.NOT_FOUND

    assert result.action == AccessAction.NOT_FOUND

    assert result.path is None


def test_ambiguous_document_requires_user(
    tmp_path,
):

    document_1 = (
        tmp_path
        / "SQL Interview Questions.docx"
    )

    document_2 = (
        tmp_path
        / "SQL Interview Questions.pdf"
    )

    document_1.write_text(
        "docx",
        encoding="utf-8",
    )

    document_2.write_text(
        "pdf",
        encoding="utf-8",
    )

    manager, _ = create_manager(tmp_path)

    result = manager.locate(
        "SQL Interview Questions"
    )

    assert result.status == AccessStatus.AMBIGUOUS

    assert result.action == AccessAction.ASK_USER

    assert result.path is None


def test_new_document_is_not_treated_as_indexed(
    tmp_path,
):

    document = tmp_path / "new_document.pdf"

    document.write_text(
        "new document",
        encoding="utf-8",
    )

    manager, registry = create_manager(tmp_path)

    result = manager.locate(
        "new_document.pdf"
    )

    assert result.status == AccessStatus.FOUND

    assert result.action == AccessAction.INDEX_DOCUMENT

    assert registry.contains(document) is False


def test_registry_state_controls_access_action(
    tmp_path,
):

    document = tmp_path / "existing_document.pdf"

    document.write_text(
        "existing document",
        encoding="utf-8",
    )

    manager, registry = create_manager(tmp_path)

    # First request: not indexed.
    first_result = manager.locate(
        "existing_document.pdf"
    )

    assert (
        first_result.action
        == AccessAction.INDEX_DOCUMENT
    )

    # Simulate successful indexing.
    registry.register(document)
    registry.mark_indexed(document)

    # Second request: already indexed.
    second_result = manager.locate(
        "existing_document.pdf"
    )

    assert (
        second_result.action
        == AccessAction.USE_EXISTING_INDEX
    )