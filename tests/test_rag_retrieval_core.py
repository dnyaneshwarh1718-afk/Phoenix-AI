from dataclasses import dataclass, field

from app.rag.context.context_builder import ContextBuilder
from app.rag.retrieval.reranker import HybridReranker
from app.rag.retrieval.rrf import reciprocal_rank_fusion


@dataclass
class Result:
    chunk_id: str
    document_id: str = "doc"
    chunk_index: int = 0
    text: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
    retrieval_method: str = "test"


def test_rrf_deduplicates_same_chunk_across_retrievers():
    dense = [Result("a", text="sql joins", score=.9), Result("b", text="indexes", score=.8)]
    bm25 = [Result("a", text="sql joins", score=4.0), Result("c", text="transactions", score=3.0)]
    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=10)
    assert [x.chunk_id for x in fused] == ["a", "b", "c"]
    assert fused[0].score > fused[1].score


def test_reranker_prioritizes_query_coverage():
    results = [
        Result("a", text="unrelated engineering content", score=.03),
        Result("b", text="SQL joins combine rows from related tables", score=.03),
    ]
    ranked = HybridReranker().rerank("SQL joins", results, limit=2)
    assert ranked[0].chunk_id == "b"


def test_context_builder_is_bounded_and_unique():
    results = [
        Result("a", text="A" * 100, score=.1),
        Result("a", text="duplicate", score=.2),
        Result("b", text="B" * 100, score=.09),
    ]
    context = ContextBuilder(max_chunks=2, max_context_chars=500, max_chunk_chars=120).build("query", results)
    assert [item.chunk_id for item in context.items] == ["a", "b"]
    assert len(context.context_text) <= 500
