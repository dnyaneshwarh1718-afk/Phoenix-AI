from __future__ import annotations

from app.rag.bm25.bm25_store import BM25Store
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.rrf import RRFResult, reciprocal_rank_fusion
from app.rag.retrieval.reranker import HybridReranker, RerankedResult
from app.rag.vector_store.qdrant_store import QdrantStore


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

    def search(
        self,
        query: str,
        limit: int = 10,
        document_id: str | None = None,
        candidate_limit: int | None = None,
    ) -> list[RerankedResult]:
        if not query or not query.strip() or limit <= 0:
            return []

        candidate_limit = max(limit, candidate_limit or limit * 4)
        query = query.strip()

        if document_id:
            dense = self.dense_retriever.search_document(query, document_id, candidate_limit)
            keyword = self.keyword_retriever.search_document(query, document_id, candidate_limit)
        else:
            dense = self.dense_retriever.search(query, candidate_limit)
            keyword = self.keyword_retriever.search(query, candidate_limit)

        fused: list[RRFResult] = reciprocal_rank_fusion(
            [dense, keyword],
            k=self.rrf_k,
            limit=candidate_limit,
        )
        return self.reranker.rerank(query, fused, limit=limit)
