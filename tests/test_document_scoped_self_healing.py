from pathlib import Path

from app.rag.vector_store.qdrant_store import QdrantStore


def test_document_scoped_qdrant_search_falls_back_to_global_filter(monkeypatch):
    store = object.__new__(QdrantStore)
    store.collection_name = "phoenix_documents"
    store.vector_size = 3

    class P:
        def __init__(self, doc, score):
            self.id = "p1"
            self.score = score
            self.payload = {"document_id": doc, "text": "Phoenix uses Qdrant", "chunk_id": "c1", "chunk_index": 0, "metadata": {}}

    class R:
        points = [P("doc-123", 0.9)]

    class C:
        def __init__(self): self.calls = 0
        def query_points(self, **kwargs):
            self.calls += 1
            if "query_filter" in kwargs:
                return type("R", (), {"points": []})()
            return R()
    store.client = C()
    result = store.search_document([0.1, 0.2, 0.3], "doc-123", limit=1)
    assert len(result) == 1
    assert result[0].payload["document_id"] == "doc-123"
