from __future__ import annotations

import os
from pathlib import Path

from app.rag.bm25.bm25_store import BM25Store
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.rrf import RRFResult, reciprocal_rank_fusion
from app.rag.retrieval.reranker import HybridReranker, RerankedResult
from app.rag.vector_store.qdrant_store import QdrantStore
from app.rag.structured.excel_analytical_retriever import ExcelAnalyticalRetriever


class HybridRetriever:
    """Dense + BM25 retrieval followed by deterministic RRF fusion and reranking."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: QdrantStore,
        bm25_store: BM25Store,
        rrf_k: int = 60,
        reranker: HybridReranker | None = None,
    ) -> None:
        self.dense_retriever = DenseRetriever(vector_store, embedding_engine)
        self.keyword_retriever = KeywordRetriever(bm25_store)
        self.rrf_k = rrf_k
        self.reranker = reranker or HybridReranker()
        self.excel_analytical_retriever = ExcelAnalyticalRetriever()

    def search(
        self,
        query: str,
        limit: int = 10,
        document_id: str | None = None,
        candidate_limit: int | None = None,
        source_path: str | None = None,
    ) -> list[RerankedResult]:
        if not query or not query.strip() or limit <= 0:
            return []

        # Structured spreadsheet questions require deterministic row/aggregation
        # execution. Dense/BM25 retrieval remains the fallback for ordinary
        # spreadsheet knowledge questions.
        if source_path:
            structured = self.excel_analytical_retriever.search(query, source_path)
            if structured:
                return structured[:limit]

        candidate_limit = max(limit, candidate_limit or limit * 4)
        query = query.strip()

        if document_id:
            dense = self.dense_retriever.search_document(query, document_id, candidate_limit)
            keyword = self.keyword_retriever.search_document(query, document_id, candidate_limit)

            # The registry and retrieval backends can drift after migrations,
            # restores, or an older indexing run. In that situation the same
            # physical document may have a different document_id in Qdrant/BM25.
            # A document-scoped query must not become a false negative merely
            # because the identity key drifted. Fall back to global candidates
            # and enforce the document boundary using the canonical source path.
            if not dense and not keyword and source_path:
                dense_candidates = self.dense_retriever.search(query, candidate_limit * 5)
                keyword_candidates = self.keyword_retriever.search(query, candidate_limit * 5)
                dense = self._filter_source_path(dense_candidates, source_path)[:candidate_limit]
                keyword = self._filter_source_path(keyword_candidates, source_path)[:candidate_limit]
        else:
            dense = self.dense_retriever.search(query, candidate_limit)
            keyword = self.keyword_retriever.search(query, candidate_limit)

        fused: list[RRFResult] = reciprocal_rank_fusion(
            [dense, keyword],
            k=self.rrf_k,
            limit=candidate_limit,
        )
        return self.reranker.rerank(query, fused, limit=limit)
    @staticmethod
    def _normalize_path(value: str) -> str:
        try:
            return os.path.normcase(str(Path(value).resolve()))
        except (OSError, RuntimeError, TypeError):
            return os.path.normcase(os.path.normpath(str(value)))

    @classmethod
    def _filter_source_path(cls, results, source_path: str):
        target = cls._normalize_path(source_path)
        matched = []
        for result in results:
            metadata = getattr(result, "metadata", {}) or {}
            candidate = metadata.get("source_path")
            if candidate and cls._normalize_path(str(candidate)) == target:
                matched.append(result)
        return matched

