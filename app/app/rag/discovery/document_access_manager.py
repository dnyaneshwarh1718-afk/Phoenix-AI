from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.rag.discovery.document_registry import (
    DocumentRegistry,
)

from app.rag.discovery.local_drive_finder import (
    DocumentSearchResult,
    LocalDriveDocumentFinder,
    MatchStatus,
)


class AccessStatus(str, Enum):
    """
    Result of a document access request.
    """

    FOUND = "FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


class AccessAction(str, Enum):
    """
    Action Phoenix should take after locating a document.
    """

    USE_EXISTING_INDEX = "USE_EXISTING_INDEX"
    INDEX_DOCUMENT = "INDEX_DOCUMENT"
    ASK_USER = "ASK_USER"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class DocumentAccessResult:
    """
    Result returned by DocumentAccessManager.
    """

    status: AccessStatus
    action: AccessAction

    query: str

    path: Path | None = None

    filename: str | None = None

    extension: str | None = None

    score: float | None = None

    message: str = ""

    finder_result: DocumentSearchResult | None = None


class DocumentAccessManager:
    """
    Coordinates:

        LocalDriveDocumentFinder
                    +
             DocumentRegistry

    Responsibilities:

        1. Locate the requested document.
        2. Handle ambiguous results.
        3. Determine whether the document is already indexed.
        4. Determine whether re-indexing is required.

    Does NOT:

        - parse documents
        - chunk documents
        - generate embeddings
        - access Qdrant
        - access BM25
        - perform retrieval
        - call an LLM
        - generate answers
    """

    def __init__(
        self,
        finder: LocalDriveDocumentFinder | None = None,
        registry: DocumentRegistry | None = None,
    ) -> None:

        self.finder = (
            finder
            if finder is not None
            else LocalDriveDocumentFinder()
        )

        self.registry = (
            registry
            if registry is not None
            else DocumentRegistry()
        )

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def locate(
        self,
        query: str,
    ) -> DocumentAccessResult:
        """
        Locate a document and determine what Phoenix
        should do next.
        """

        finder_result = self.finder.find(query)

        # ------------------------------------------------------
        # DOCUMENT NOT FOUND
        # ------------------------------------------------------

        if finder_result.status == MatchStatus.NOT_FOUND:

            return DocumentAccessResult(
                status=AccessStatus.NOT_FOUND,
                action=AccessAction.NOT_FOUND,
                query=query,
                message=finder_result.message,
                finder_result=finder_result,
            )

        # ------------------------------------------------------
        # MULTIPLE DOCUMENTS
        # ------------------------------------------------------

        if finder_result.status == MatchStatus.AMBIGUOUS:

            return DocumentAccessResult(
                status=AccessStatus.AMBIGUOUS,
                action=AccessAction.ASK_USER,
                query=query,
                message=(
                    "Multiple documents match the request. "
                    "User selection is required."
                ),
                finder_result=finder_result,
            )

        # ------------------------------------------------------
        # DOCUMENT FOUND
        # ------------------------------------------------------

        path = finder_result.path

        if path is None:

            return DocumentAccessResult(
                status=AccessStatus.NOT_FOUND,
                action=AccessAction.NOT_FOUND,
                query=query,
                message=(
                    "Finder reported FOUND but returned "
                    "no document path."
                ),
                finder_result=finder_result,
            )

        # ------------------------------------------------------
        # CHECK REGISTRY
        # ------------------------------------------------------

        if self.registry.is_indexed(path):

            return DocumentAccessResult(
                status=AccessStatus.FOUND,
                action=AccessAction.USE_EXISTING_INDEX,
                query=query,
                path=path,
                filename=finder_result.filename,
                extension=finder_result.extension,
                score=finder_result.score,
                message=(
                    f"Document is already indexed: {path}"
                ),
                finder_result=finder_result,
            )

        # ------------------------------------------------------
        # NEW OR MODIFIED DOCUMENT
        # ------------------------------------------------------

        return DocumentAccessResult(
            status=AccessStatus.FOUND,
            action=AccessAction.INDEX_DOCUMENT,
            query=query,
            path=path,
            filename=finder_result.filename,
            extension=finder_result.extension,
            score=finder_result.score,
            message=(
                f"Document requires indexing: {path}"
            ),
            finder_result=finder_result,
        )