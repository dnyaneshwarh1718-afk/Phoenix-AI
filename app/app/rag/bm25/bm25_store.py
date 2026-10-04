from __future__ import annotations

import json
import re
from pathlib import Path
from threading import RLock

from rank_bm25 import BM25Okapi

from app.rag.models import BM25RetrievalResult, DocumentChunk


class BM25Store:
    """
    Persistent local BM25 index.

    The index is rebuilt from persisted chunks on startup. This avoids the
    previous failure mode where BM25 silently became empty after a Phoenix
    restart while Qdrant still contained vectors.
    """

    def __init__(self, storage_path: str = "data/bm25_chunks.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: list[DocumentChunk] = []
        self.index: BM25Okapi | None = None
        self._lock = RLock()
        self._load()
        self._rebuild_index()

    def build(self, chunks: list[DocumentChunk]) -> None:
        with self._lock:
            self.chunks = list(chunks or [])
            self._rebuild_index()
            self._save()

    def add_chunks(self, chunks: list[DocumentChunk], document_id: str | None = None) -> None:
        incoming = list(chunks or [])
        if not incoming:
            return
        with self._lock:
            doc_ids = {c.document_id for c in incoming if getattr(c, "document_id", None)}
            if document_id:
                doc_ids.add(document_id)
            # Replace an existing document atomically. Chunk IDs are deterministic,
            # so reindexing must not create duplicate BM25 entries.
            self.chunks = [c for c in self.chunks if c.document_id not in doc_ids]
            self.chunks.extend(incoming)
            self._rebuild_index()
            self._save()

    def remove_document(self, document_id: str) -> None:
        if not document_id:
            return
        with self._lock:
            self.chunks = [c for c in self.chunks if c.document_id != document_id]
            self._rebuild_index()
            self._save()

    def has_document(self, document_id: str) -> bool:
        return any(c.document_id == document_id for c in self.chunks)

    def document_chunk_count(self, document_id: str) -> int:
        return sum(c.document_id == document_id for c in self.chunks)

    def count(self) -> int:
        return len(self.chunks)

    def search(self, query: str, limit: int = 10) -> list[BM25RetrievalResult]:
        return self._search_chunks(query, self.chunks, limit)

    def search_document(self, query: str, document_id: str, limit: int = 10) -> list[BM25RetrievalResult]:
        if not document_id:
            return []
        chunks = [c for c in self.chunks if c.document_id == document_id]
        return self._search_chunks(query, chunks, limit)

    def health_check(self) -> bool:
        return True

    def _rebuild_index(self) -> None:
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.index = BM25Okapi(tokenized) if tokenized else None

    def _search_chunks(
        self,
        query: str,
        chunks: list[DocumentChunk],
        limit: int,
    ) -> list[BM25RetrievalResult]:
        if not query or not query.strip() or limit <= 0 or not chunks:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # For document-scoped searches, build a small temporary index so that
        # scores remain local to the requested document.
        if chunks is self.chunks:
            index = self.index
        else:
            index = BM25Okapi([self._tokenize(c.text) for c in chunks])
        if index is None:
            return []

        scores = index.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: (-float(pair[1]), chunks[pair[0]].chunk_id),
        )
        return [
            BM25RetrievalResult(chunk=chunks[i], score=float(score), retrieval_method="bm25")
            for i, score in ranked[:limit]
            if float(score) > 0.0
        ]

    def _save(self) -> None:
        payload = []
        for chunk in self.chunks:
            payload.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            )
        tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.storage_path)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load BM25 store: {self.storage_path}") from exc
        if not isinstance(raw, list):
            raise RuntimeError(f"Invalid BM25 store format: {self.storage_path}")
        loaded: list[DocumentChunk] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                loaded.append(
                    DocumentChunk(
                        chunk_id=str(item["chunk_id"]),
                        document_id=str(item["document_id"]),
                        text=str(item["text"]),
                        chunk_index=int(item["chunk_index"]),
                        metadata=dict(item.get("metadata") or {}),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self.chunks = loaded

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)
