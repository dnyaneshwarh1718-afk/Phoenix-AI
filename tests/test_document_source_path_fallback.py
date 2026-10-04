from app.rag.retrieval.hybrid_retriever import HybridRetriever


def test_source_path_fallback_filters_candidates_without_cross_document_leakage():
    class Result:
        def __init__(self, path, chunk_id):
            self.metadata = {"source_path": path}
            self.chunk_id = chunk_id

    results = [
        Result(r"R:\Phoenix-AI-main\evaluation\corpus\phoenix_architecture.txt", "good"),
        Result(r"R:\other\secret.txt", "bad"),
    ]
    matched = HybridRetriever._filter_source_path(
        results,
        r"R:\Phoenix-AI-main\evaluation\corpus\phoenix_architecture.txt",
    )
    assert [x.chunk_id for x in matched] == ["good"]
