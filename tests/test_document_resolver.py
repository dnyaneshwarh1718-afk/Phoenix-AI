import tempfile
from pathlib import Path

from app.rag.registry.document_registry import DocumentRegistry
from app.rag.document_resolution.document_resolver import (
    DocumentResolver,
)


def print_result(
    title: str,
    result,
) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Found: {result.found}")
    print(f"Match Type: {result.match_type}")
    print(f"Message: {result.message}")

    if result.document:

        print(
            f"Document ID: "
            f"{result.document.document_id}"
        )

        print(
            f"File Name: "
            f"{result.document.file_name}"
        )

        print(
            f"Source Path: "
            f"{result.document.source_path}"
        )

        print(
            f"Chunks: "
            f"{result.document.chunk_count}"
        )


def main():

    print("=" * 70)
    print("PHOENIX AI - DOCUMENT RESOLVER TEST")
    print("=" * 70)

    # ==================================================
    # TEMPORARY REGISTRY
    # ==================================================

    with tempfile.TemporaryDirectory() as temp_dir:

        registry_path = (
            Path(temp_dir)
            / "document_registry.json"
        )

        registry = DocumentRegistry(
            registry_path=str(
                registry_path
            )
        )

        # ==================================================
        # REGISTER TEST DOCUMENTS
        # ==================================================

        registry.register(
            document_id="doc-001",
            file_name="practical_statistics.pdf",
            source_path=(
                r"R:\Documents\practical_statistics.pdf"
            ),
            file_type=".pdf",
            chunk_count=42,
        )

        registry.register(
            document_id="doc-002",
            file_name="machine_learning.txt",
            source_path=(
                r"R:\Documents\machine_learning.txt"
            ),
            file_type=".txt",
            chunk_count=15,
        )

        # ==================================================
        # RESOLVER
        # ==================================================

        resolver = DocumentResolver(
            document_registry=registry
        )

        # ==================================================
        # TEST 1 — DOCUMENT ID
        # ==================================================

        result = resolver.resolve(
            "doc-001"
        )

        print_result(
            "TEST 1 - DOCUMENT ID",
            result,
        )

        assert result.found
        assert (
            result.document.document_id
            == "doc-001"
        )

        # ==================================================
        # TEST 2 — EXACT FILENAME
        # ==================================================

        result = resolver.resolve(
            "practical_statistics.pdf"
        )

        print_result(
            "TEST 2 - EXACT FILENAME",
            result,
        )

        assert result.found
        assert (
            result.document.file_name
            == "practical_statistics.pdf"
        )

        # ==================================================
        # TEST 3 — CASE INSENSITIVE
        # ==================================================

        result = resolver.resolve(
            "PRACTICAL_STATISTICS.PDF"
        )

        print_result(
            "TEST 3 - CASE-INSENSITIVE FILENAME",
            result,
        )

        assert result.found
        assert (
            result.document.document_id
            == "doc-001"
        )

        # ==================================================
        # TEST 4 — SOURCE PATH
        # ==================================================

        result = resolver.resolve(
            r"R:\Documents\machine_learning.txt"
        )

        print_result(
            "TEST 4 - SOURCE PATH",
            result,
        )

        assert result.found
        assert (
            result.document.document_id
            == "doc-002"
        )

        # ==================================================
        # TEST 5 — DIRECT FILENAME METHOD
        # ==================================================

        result = resolver.resolve_by_filename(
            "machine_learning.txt"
        )

        print_result(
            "TEST 5 - RESOLVE BY FILENAME",
            result,
        )

        assert result.found

        # ==================================================
        # TEST 6 — DIRECT ID METHOD
        # ==================================================

        result = resolver.resolve_by_id(
            "doc-001"
        )

        print_result(
            "TEST 6 - RESOLVE BY ID",
            result,
        )

        assert result.found

        # ==================================================
        # TEST 7 — DOCUMENT NOT FOUND
        # ==================================================

        result = resolver.resolve(
            "unknown_document.pdf"
        )

        print_result(
            "TEST 7 - DOCUMENT NOT FOUND",
            result,
        )

        assert not result.found

        # ==================================================
        # FINAL STATUS
        # ==================================================

        print("\n" + "=" * 70)
        print(
            "DOCUMENT RESOLVER TEST STATUS: PASSED"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()