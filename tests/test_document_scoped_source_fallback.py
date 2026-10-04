from pathlib import Path

from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbedding:
    def embed_query(self, query):
        return [0.1, 0.2]


class FakePoint:
    def __init__(self):
        self.id = "chunk-1"
        self.score = 0.9
        self.payload = {
            "chunk_id": "chunk-1",
            "document_id": "canonical-doc-id",
            "chunk_index": 0,
            "text": "Phoenix uses Qdrant.",
            "metadata": {"source_path": str(Path("R:/Phoenix-AI-main/evaluation/corpus/phoenix_architecture.txt").resolve())},
        }


class FakeVectorStore:
    def search_document(self, query_vector, document_id, limit=5, source_path=None):
        # Simulate a legacy index whose document_id filter misses the point.
        return [FakePoint()] if source_path else []

    def search(self, query_vector, limit=5):
        return []


class FakeBM25:
    def search_document(self, query, document_id, limit=5, source_path=None):
        return []

    def search(self, query, limit=5):
        return []


def test_document_scoped_retrieval_accepts_legacy_id_mismatch_via_source_path():
    dense = DenseRetriever(FakeVectorStore(), FakeEmbedding())
    results = dense.search_document(
        "What vector database is used?",
        "stale-doc-id",
        5,
        source_path="R:/Phoenix-AI-main/evaluation/corpus/phoenix_architecture.txt",
    )
    assert len(results) == 1
    assert results[0].text == "Phoenix uses Qdrant."
