from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DocumentIdentity:
    """
    Stable filesystem identity information for a document.

    This is metadata only.
    No document contents are stored here.
    """

    path: Path
    filename: str
    extension: str
    file_size: int
    modified_time_ns: int


@dataclass
class DocumentRecord:
    """
    Registry record describing the indexing state of a document.
    """

    identity: DocumentIdentity
    indexed: bool = False


class DocumentRegistry:
    """
    In-memory registry for documents discovered by Phoenix AI.

    Responsibilities:
        - Identify documents using filesystem metadata.
        - Track whether a document has been indexed.
        - Detect whether an indexed document has changed.

    Does NOT:
        - read document contents
        - parse documents
        - chunk documents
        - generate embeddings
        - access Qdrant
        - access BM25
        - perform retrieval
        - call an LLM
    """

    def __init__(self) -> None:
        self._records: Dict[str, DocumentRecord] = {}

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def register(
        self,
        path: str | Path,
        indexed: bool = False,
    ) -> DocumentRecord:
        """
        Register a document in the registry.

        If the document already exists, its metadata is refreshed.
        """

        path = Path(path)

        identity = self._build_identity(path)

        key = self._make_key(path)

        record = DocumentRecord(
            identity=identity,
            indexed=indexed,
        )

        self._records[key] = record

        return record

    def get(
        self,
        path: str | Path,
    ) -> DocumentRecord | None:
        """
        Return the registry record for a document.

        Returns None if the document is not registered.
        """

        key = self._make_key(path)

        return self._records.get(key)

    def contains(
        self,
        path: str | Path,
    ) -> bool:
        """
        Return True if the document exists in the registry.
        """

        return self.get(path) is not None

    def mark_indexed(
        self,
        path: str | Path,
    ) -> DocumentRecord:
        """
        Mark a document as successfully indexed.

        The document must already exist in the registry.
        """

        path = Path(path)

        record = self.get(path)

        if record is None:
            record = self.register(
                path=path,
                indexed=True,
            )
            return record

        updated_record = DocumentRecord(
            identity=self._build_identity(path),
            indexed=True,
        )

        self._records[
            self._make_key(path)
        ] = updated_record

        return updated_record

    def mark_not_indexed(
        self,
        path: str | Path,
    ) -> DocumentRecord:
        """
        Mark a document as not indexed.
        """

        path = Path(path)

        record = self.get(path)

        if record is None:
            return self.register(
                path=path,
                indexed=False,
            )

        updated_record = DocumentRecord(
            identity=self._build_identity(path),
            indexed=False,
        )

        self._records[
            self._make_key(path)
        ] = updated_record

        return updated_record

    def is_indexed(
        self,
        path: str | Path,
    ) -> bool:
        """
        Return True only when the document is registered
        and currently indexed.
        """

        record = self.get(path)

        if record is None:
            return False

        if not self._identity_matches_current_file(
            record.identity
        ):
            return False

        return record.indexed

    def needs_reindex(
        self,
        path: str | Path,
    ) -> bool:
        """
        Determine whether a document needs indexing.

        True when:
            - document is not registered
            - document is not indexed
            - document metadata has changed
        """

        path = Path(path)

        record = self.get(path)

        if record is None:
            return True

        if not record.indexed:
            return True

        return not self._identity_matches_current_file(
            record.identity
        )

    def remove(
        self,
        path: str | Path,
    ) -> bool:
        """
        Remove a document from the registry.

        Returns True when a record was removed.
        """

        key = self._make_key(path)

        return self._records.pop(
            key,
            None,
        ) is not None

    def count(self) -> int:
        """
        Return the number of registered documents.
        """

        return len(self._records)

    # ==========================================================
    # INTERNAL METHODS
    # ==========================================================

    @staticmethod
    def _make_key(
        path: str | Path,
    ) -> str:
        """
        Create a normalized filesystem key.

        Windows paths are case-insensitive, so the key is
        normalized to lowercase.
        """

        return str(
            Path(path)
            .resolve()
        ).lower()

    @staticmethod
    def _build_identity(
        path: Path,
    ) -> DocumentIdentity:
        """
        Read filesystem metadata required for identity tracking.
        """

        stat = path.stat()

        return DocumentIdentity(
            path=path,
            filename=path.name,
            extension=path.suffix.lower(),
            file_size=stat.st_size,
            modified_time_ns=stat.st_mtime_ns,
        )

    @staticmethod
    def _identity_matches_current_file(
        identity: DocumentIdentity,
    ) -> bool:
        """
        Determine whether the filesystem metadata still matches
        the metadata recorded when the document was indexed.
        """

        path = identity.path

        if not path.exists():
            return False

        try:
            current_stat = path.stat()
        except OSError:
            return False

        return (
            current_stat.st_size
            == identity.file_size
            and current_stat.st_mtime_ns
            == identity.modified_time_ns
        )