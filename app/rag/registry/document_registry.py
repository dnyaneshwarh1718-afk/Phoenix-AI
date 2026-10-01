from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Optional


@dataclass
class DocumentRecord:
    """Persistent metadata describing an indexed document."""

    document_id: str
    file_name: str
    source_path: str
    file_type: str
    chunk_count: int
    indexed_at: str
    status: str = "indexed"
    file_size: int = 0
    modified_time_ns: int = 0


class DocumentRegistry:
    """
    Persistent registry for Phoenix AI documents.

    Responsibilities:

    - Register indexed documents
    - Find documents by ID
    - Find documents by filename
    - List indexed documents
    - Remove document records
    - Persist registry metadata to disk

    The registry does NOT store document chunks.

    Qdrant:
        Stores vectors + chunks

    BM25:
        Stores keyword index

    Registry:
        Stores document identity + metadata
    """

    def __init__(
        self,
        registry_path: str = "data/document_registry.json",
    ) -> None:

        self.registry_path = Path(
            registry_path
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._documents: dict[
            str,
            DocumentRecord,
        ] = {}

        self._load()

    # ==========================================================
    # LOAD
    # ==========================================================

    def _load(self) -> None:
        """
        Load registry from disk.
        """

        if not self.registry_path.exists():
            return

        try:

            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            for document_id, record in data.items():
                if not isinstance(record, dict):
                    continue
                # Backward compatible with registries created before file
                # fingerprint fields were introduced.
                record.setdefault("file_size", 0)
                record.setdefault("modified_time_ns", 0)
                self._documents[document_id] = DocumentRecord(**record)

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):

            raise RuntimeError(
                f"Invalid document registry: "
                f"{self.registry_path}"
            )

    # ==========================================================
    # SAVE
    # ==========================================================

    def _save(self) -> None:
        """
        Persist registry to disk.
        """

        data = {
            document_id: asdict(record)
            for document_id, record
            in self._documents.items()
        }

        with self.registry_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # ==========================================================
    # REGISTER
    # ==========================================================

    def register(
        self,
        document_id: str,
        file_name: str,
        source_path: str,
        file_type: str,
        chunk_count: int,
        status: str = "indexed",
    ) -> DocumentRecord:

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        if not file_name:
            raise ValueError(
                "file_name cannot be empty."
            )

        if chunk_count < 0:
            raise ValueError(
                "chunk_count cannot be negative."
            )

        resolved = Path(source_path).resolve()
        stat = resolved.stat() if resolved.exists() else None
        record = DocumentRecord(
            document_id=document_id,
            file_name=file_name,
            source_path=str(resolved),
            file_type=file_type,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            file_size=stat.st_size if stat else 0,
            modified_time_ns=stat.st_mtime_ns if stat else 0,
        )

        self._documents[
            document_id
        ] = record

        self._save()

        return record

    # ==========================================================
    # GET BY ID
    # ==========================================================

    def get(self, document_id_or_path: str) -> Optional[DocumentRecord]:
        """Resolve a record by document ID or source path."""
        if not document_id_or_path:
            return None
        direct = self._documents.get(str(document_id_or_path))
        if direct is not None:
            return direct
        target = str(Path(document_id_or_path).resolve()).lower()
        for record in self._documents.values():
            if str(Path(record.source_path).resolve()).lower() == target:
                return record
        return None

    def is_indexed(self, path: str | Path) -> bool:
        """Return True only if the file still matches its indexed fingerprint."""
        record = self.find_by_path(str(path))
        if record is None or record.status != "indexed":
            return False
        target = Path(record.source_path)
        if not target.is_file():
            return False
        stat = target.stat()
        # Legacy records with zero fingerprint are accepted once and refreshed
        # by the next successful indexing operation.
        if record.file_size == 0 and record.modified_time_ns == 0:
            return True
        return stat.st_size == record.file_size and stat.st_mtime_ns == record.modified_time_ns

    # ==========================================================
    # FIND BY FILE NAME
    # ==========================================================

    def find_by_name(
        self,
        file_name: str,
    ) -> Optional[DocumentRecord]:

        target = file_name.strip().lower()

        for record in self._documents.values():

            if (
                record.file_name.lower()
                == target
            ):
                return record

        return None

    # ==========================================================
    # FIND BY PATH
    # ==========================================================

    def find_by_path(
        self,
        source_path: str,
    ) -> Optional[DocumentRecord]:

        target = str(
            Path(source_path).resolve()
        ).lower()

        for record in self._documents.values():

            if (
                str(
                    Path(
                        record.source_path
                    ).resolve()
                ).lower()
                == target
            ):
                return record

        return None

    # ==========================================================
    # LIST
    # ==========================================================

    def list_documents(
        self,
    ) -> list[DocumentRecord]:

        return list(
            self._documents.values()
        )

    # ==========================================================
    # REMOVE
    # ==========================================================

    def remove(
        self,
        document_id: str,
    ) -> bool:

        if (
            document_id
            not in self._documents
        ):
            return False

        del self._documents[
            document_id
        ]

        self._save()

        return True

    # ==========================================================
    # EXISTS
    # ==========================================================

    def exists(
        self,
        document_id: str,
    ) -> bool:

        return (
            document_id
            in self._documents
        )

    # ==========================================================
    # COUNT
    # ==========================================================

    def count(self) -> int:

        return len(
            self._documents
        )