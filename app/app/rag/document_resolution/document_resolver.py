from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.rag.registry.document_registry import (
    DocumentRegistry,
    DocumentRecord,
)


@dataclass
class DocumentResolutionResult:
    """
    Result returned when Phoenix attempts to resolve
    a user-provided document reference.
    """

    found: bool
    document: Optional[DocumentRecord] = None
    match_type: Optional[str] = None
    message: str = ""


class DocumentResolver:
    """
    Resolves a user document reference to a registered
    Phoenix document.

    Resolution priority:

        1. Document ID
        2. Exact filename
        3. Exact source path
        4. Case-insensitive filename

    This component does NOT perform retrieval.

    Its only responsibility is:

        User document reference
                ↓
        DocumentRegistry
                ↓
        DocumentRecord
    """

    def __init__(
        self,
        document_registry: DocumentRegistry,
    ) -> None:

        self.document_registry = document_registry

    # ==========================================================
    # RESOLVE
    # ==========================================================

    def resolve(
        self,
        reference: str,
    ) -> DocumentResolutionResult:
        """
        Resolve a document reference.

        Examples:

            "practical_statistics.pdf"

            "Practical_Statistics.pdf"

            "doc-001"

            "R:\\Documents\\practical_statistics.pdf"
        """

        if not reference or not reference.strip():

            return DocumentResolutionResult(
                found=False,
                message="Document reference cannot be empty.",
            )

        reference = reference.strip()

        # ------------------------------------------------------
        # 1. DOCUMENT ID
        # ------------------------------------------------------

        document = self.document_registry.get(
            reference
        )

        if document is not None:

            return DocumentResolutionResult(
                found=True,
                document=document,
                match_type="document_id",
                message=(
                    f"Document resolved by ID: "
                    f"{document.file_name}"
                ),
            )

        # ------------------------------------------------------
        # 2. EXACT SOURCE PATH
        # ------------------------------------------------------

        path = Path(reference)

        if path.exists() and path.is_file():

            document = (
                self.document_registry.find_by_path(
                    reference
                )
            )

            if document is not None:

                return DocumentResolutionResult(
                    found=True,
                    document=document,
                    match_type="source_path",
                    message=(
                        f"Document resolved by source path: "
                        f"{document.file_name}"
                    ),
                )

        # Even if the file no longer exists locally,
        # the registry may still contain its historical path.

        document = (
            self.document_registry.find_by_path(
                reference
            )
        )

        if document is not None:

            return DocumentResolutionResult(
                found=True,
                document=document,
                match_type="source_path",
                message=(
                    f"Document resolved by registered "
                    f"source path: {document.file_name}"
                ),
            )

        # ------------------------------------------------------
        # 3. EXACT FILENAME
        # ------------------------------------------------------

        document = (
            self.document_registry.find_by_name(
                reference
            )
        )

        if document is not None:

            return DocumentResolutionResult(
                found=True,
                document=document,
                match_type="filename",
                message=(
                    f"Document resolved by filename: "
                    f"{document.file_name}"
                ),
            )

        # ------------------------------------------------------
        # 4. CASE-INSENSITIVE FILENAME
        # ------------------------------------------------------

        reference_lower = reference.lower()

        matches: list[DocumentRecord] = []

        for record in (
            self.document_registry.list_documents()
        ):

            if (
                record.file_name.lower()
                == reference_lower
            ):

                matches.append(record)

        # ------------------------------------------------------
        # SINGLE MATCH
        # ------------------------------------------------------

        if len(matches) == 1:

            document = matches[0]

            return DocumentResolutionResult(
                found=True,
                document=document,
                match_type="case_insensitive_filename",
                message=(
                    f"Document resolved by "
                    f"case-insensitive filename: "
                    f"{document.file_name}"
                ),
            )

        # ------------------------------------------------------
        # MULTIPLE MATCHES
        # ------------------------------------------------------

        if len(matches) > 1:

            return DocumentResolutionResult(
                found=False,
                message=(
                    "Multiple documents matched "
                    f"'{reference}'. "
                    "Phoenix must ask the user to "
                    "select the correct document."
                ),
            )

        # ------------------------------------------------------
        # NOT FOUND
        # ------------------------------------------------------

        return DocumentResolutionResult(
            found=False,
            message=(
                f"No registered document found "
                f"for '{reference}'."
            ),
        )

    # ==========================================================
    # RESOLVE BY DOCUMENT ID
    # ==========================================================

    def resolve_by_id(
        self,
        document_id: str,
    ) -> DocumentResolutionResult:
        """
        Resolve a document directly using its
        Phoenix document ID.
        """

        if not document_id or not document_id.strip():

            return DocumentResolutionResult(
                found=False,
                message="Document ID cannot be empty.",
            )

        document = self.document_registry.get(
            document_id.strip()
        )

        if document is None:

            return DocumentResolutionResult(
                found=False,
                message=(
                    f"No document found with ID "
                    f"'{document_id}'."
                ),
            )

        return DocumentResolutionResult(
            found=True,
            document=document,
            match_type="document_id",
            message=(
                f"Document resolved: "
                f"{document.file_name}"
            ),
        )

    # ==========================================================
    # RESOLVE BY FILENAME
    # ==========================================================

    def resolve_by_filename(
        self,
        file_name: str,
    ) -> DocumentResolutionResult:
        """
        Resolve a document using its filename.
        """

        if not file_name or not file_name.strip():

            return DocumentResolutionResult(
                found=False,
                message="Filename cannot be empty.",
            )

        document = (
            self.document_registry.find_by_name(
                file_name.strip()
            )
        )

        if document is None:

            return DocumentResolutionResult(
                found=False,
                message=(
                    f"No registered document found "
                    f"with filename '{file_name}'."
                ),
            )

        return DocumentResolutionResult(
            found=True,
            document=document,
            match_type="filename",
            message=(
                f"Document resolved: "
                f"{document.file_name}"
            ),
        )