from __future__ import annotations

from app.rag.bm25.bm25_store import BM25Store


class KeywordRetriever:
    """Thin adapter around the persistent BM25 store."""

    def __init__(self, bm25_store: BM25Store) -> None:
        self.bm25_store = bm25_store

    def search(self, query: str, limit: int = 5):
        if not query or not query.strip() or limit <= 0:
            return []
        return self.bm25_store.search(query.strip(), limit)

    def search_document(self, query: str, document_id: str, limit: int = 5):
        if not query or not query.strip() or not document_id or limit <= 0:
            return []
        return self.bm25_store.search_document(query.strip(), document_id.strip(), limit)
