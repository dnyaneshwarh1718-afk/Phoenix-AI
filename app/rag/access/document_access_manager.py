from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.rag.discovery.document_discovery import DiscoveredDocument, DocumentDiscovery
from app.rag.registry.document_registry import DocumentRecord, DocumentRegistry


@dataclass
class DocumentAccessResult:
    found: bool
    ready: bool
    document: DocumentRecord | None = None
    matched_path: str | None = None
    match_type: str | None = None
    status: str = "not_found"
    message: str = ""
    candidates: list[DiscoveredDocument] = field(default_factory=list)
    indexing_result: object | None = None


class DocumentAccessManager:
    """Resolve a document reference and ensure a usable index exists."""

    def __init__(
        self,
        document_registry: DocumentRegistry,
        document_discovery: DocumentDiscovery,
        document_indexer,
        default_search_roots: list[str] | None = None,
    ) -> None:
        self.document_registry = document_registry
        self.document_discovery = document_discovery
        self.document_indexer = document_indexer
        self.default_search_roots = default_search_roots or ["R:\\"]

    def prepare_document(
        self,
        reference: str,
        search_roots: list[str] | None = None,
        *,
        auto_index: bool = True,
        candidate_limit: int = 10,
    ) -> DocumentAccessResult:
        if not reference or not reference.strip():
            return DocumentAccessResult(False, False, status="invalid_reference", message="Document reference cannot be empty.")

        reference = reference.strip()
        roots = search_roots or self.default_search_roots

        record = self._resolve_registered(reference)
        if record is not None:
            if self.document_registry.is_indexed(record.source_path):
                return self._ready(record, "registry", f"Document resolved from registry: {record.file_name}")
            # Existing record but stale/unindexed: continue to reindex it.
            source = Path(record.source_path)
            if source.is_file():
                return self._index_selected(source, "registry_stale", auto_index)

        direct = Path(reference)
        if direct.is_file():
            return self._index_selected(direct, "source_path", auto_index)

        candidates: list[DiscoveredDocument] = []
        for root in roots:
            try:
                candidates.extend(self.document_discovery.search(root, reference, candidate_limit))
            except (FileNotFoundError, ValueError):
                continue
        candidates = self._dedupe(candidates)

        if not candidates:
            return DocumentAccessResult(False, False, status="not_found", message=f"No supported document found for '{reference}'.")

        indexed_candidates = [
            (c, self.document_registry.find_by_path(str(c.path)))
            for c in candidates
        ]
        valid_indexed = [
            (c, r) for c, r in indexed_candidates
            if r is not None and self.document_registry.is_indexed(str(c.path))
        ]
        if len(valid_indexed) == 1:
            c, r = valid_indexed[0]
            return self._ready(r, "registered_discovery", f"Document found and already indexed: {r.file_name}", candidates)

        if len(candidates) > 1:
            return DocumentAccessResult(
                True,
                False,
                status="ambiguous",
                match_type="filesystem_discovery",
                message=f"Multiple documents matched '{reference}'. Phoenix must ask the user to select one.",
                candidates=candidates,
            )

        return self._index_selected(candidates[0].path, "filesystem_discovery", auto_index, candidates)

    def _resolve_registered(self, reference: str) -> DocumentRecord | None:
        record = self.document_registry.get(reference)
        if record:
            return record
        record = self.document_registry.find_by_name(reference)
        if record:
            return record
        record = self.document_registry.find_by_path(reference)
        if record:
            return record
        target = reference.lower()
        matches = [
            r for r in self.document_registry.list_documents()
            if Path(r.file_name).stem.lower() == Path(target).stem.lower()
        ]
        return matches[0] if len(matches) == 1 else None

    def _index_selected(self, path: Path, match_type: str, auto_index: bool, candidates=None) -> DocumentAccessResult:
        existing = self.document_registry.find_by_path(str(path))
        if existing and self.document_registry.is_indexed(str(path)):
            return self._ready(existing, match_type, f"Document already indexed: {existing.file_name}", candidates or [])
        if not auto_index:
            return DocumentAccessResult(True, False, existing, str(path.resolve()), match_type, "discovered_not_indexed", f"Document found but not indexed: {path.name}", candidates or [])

        result = self.document_indexer.index_document(str(path))
        if result.status.name not in {"INDEXED", "REINDEXED", "ALREADY_INDEXED"}:
            return DocumentAccessResult(True, False, existing, str(path.resolve()), match_type, "index_failed", result.message, candidates or [], result)

        record = self.document_registry.get(result.document_id) if result.document_id else self.document_registry.find_by_path(str(path))
        ready = record is not None and self.document_registry.is_indexed(str(path))
        return DocumentAccessResult(True, ready, record, str(path.resolve()), match_type, "indexed" if ready else "index_failed", result.message, candidates or [], result)

    @staticmethod
    def _ready(record: DocumentRecord, match_type: str, message: str, candidates=None) -> DocumentAccessResult:
        return DocumentAccessResult(True, True, record, record.source_path, match_type, "ready", message, candidates or [])

    @staticmethod
    def _dedupe(candidates: list[DiscoveredDocument]) -> list[DiscoveredDocument]:
        seen: set[str] = set()
        result: list[DiscoveredDocument] = []
        for candidate in candidates:
            key = str(candidate.path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                result.append(candidate)
        return result
