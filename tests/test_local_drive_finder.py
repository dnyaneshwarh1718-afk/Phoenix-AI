from pathlib import Path

from app.rag.discovery.local_drive_finder import (
    LocalDriveDocumentFinder,
    MatchStatus,
    MatchType,
)


def create_file(path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "test document",
        encoding="utf-8",
    )


def test_exact_filename(tmp_path):

    r_drive = tmp_path / "R"
    downloads = tmp_path / "Downloads"

    target = (
        r_drive
        / "Books"
        / "practical_statistics.pdf"
    )

    create_file(target)

    finder = LocalDriveDocumentFinder(
        primary_root=r_drive,
        downloads_root=downloads,
    )

    result = finder.find(
        "practical_statistics.pdf"
    )

    assert result.status == MatchStatus.FOUND
    assert (
        result.match_type
        == MatchType.EXACT_FILENAME
    )
    assert result.path == target


def test_exact_stem(tmp_path):

    r_drive = tmp_path / "R"

    target = (
        r_drive
        / "Books"
        / "practical_statistics.pdf"
    )

    create_file(target)

    finder = LocalDriveDocumentFinder(
        primary_root=r_drive,
        downloads_root=tmp_path / "Downloads",
    )

    result = finder.find(
        "practical_statistics"
    )

    assert result.status == MatchStatus.FOUND
    assert (
        result.match_type
        == MatchType.EXACT_STEM
    )
    assert result.path == target


def test_unrelated_token_overlap_is_not_a_match(
    tmp_path,
):

    r_drive = tmp_path / "R"

    unrelated = (
        r_drive
        / "Books"
        / (
            "A Practical Guide to "
            "Support Vector Classification.pdf"
        )
    )

    create_file(unrelated)

    finder = LocalDriveDocumentFinder(
        primary_root=r_drive,
        downloads_root=tmp_path / "Downloads",
    )

    result = finder.find(
        "practical_statistics.pdf"
    )

    assert result.status == MatchStatus.NOT_FOUND


def test_nonexistent_document_returns_not_found(
    tmp_path,
):

    r_drive = tmp_path / "R"

    create_file(
        r_drive / "random_document.pdf"
    )

    finder = LocalDriveDocumentFinder(
        primary_root=r_drive,
        downloads_root=tmp_path / "Downloads",
    )

    result = finder.find(
        "phoenix_document_that_does_not_exist"
    )

    assert result.status == MatchStatus.NOT_FOUND


def test_downloads_is_fallback(tmp_path):

    r_drive = tmp_path / "R"
    downloads = tmp_path / "Downloads"

    target = (
        downloads
        / "practical_statistics.pdf"
    )

    create_file(target)

    finder = LocalDriveDocumentFinder(
        primary_root=r_drive,
        downloads_root=downloads,
    )

    result = finder.find(
        "practical_statistics.pdf"
    )

    assert result.status == MatchStatus.FOUND
    assert result.path == target


def test_r_drive_has_priority_over_downloads(
    tmp_path,
):

    r_drive = tmp_path / "R"
    downloads = tmp_path / "Downloads"

    r_target = (
        r_drive
        / "Books"
        / "practical_statistics.pdf"
    )

    download_target = (
        downloads
        / "practical_statistics.pdf"
    )

    create_file(r_target)
    create_file(download_target)

    finder = LocalDriveDocumentFinder(
        primary_root=r_drive,
        downloads_root=downloads,
    )

    result = finder.find(
        "practical_statistics.pdf"
    )

    assert result.status == MatchStatus.FOUND
    assert result.path == r_target