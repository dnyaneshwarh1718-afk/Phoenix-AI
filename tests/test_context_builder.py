from app.rag.document_manager import DocumentManager
from app.rag.chunking.chunker import DocumentChunker

from app.rag.embeddings.embedding_engine import (
    EmbeddingEngine,
)

from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)

from app.rag.bm25.bm25_store import (
    BM25Store,
)

from app.rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from app.rag.retrieval.keyword_retriever import (
    KeywordRetriever,
)

from app.rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)

from app.rag.context.context_builder import (
    ContextBuilder,
)


def main():

    file_path = (
        "tests/rag_test_corpus.txt"
    )

    print("=" * 70)
    print(
        "PHOENIX AI - CONTEXT BUILDER TEST"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load document
    # --------------------------------------------------

    manager = DocumentManager()

    document = manager.load_document(
        file_path
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk(
        document
    )

    print(
        f"\nDocument: {document.file_name}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    bm25 = BM25Store()

    bm25.build(
        chunks
    )

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    embedding_engine = EmbeddingEngine(
        model_name="nomic-embed-text"
    )

    vector_store = QdrantStore(
        collection_name="phoenix_documents"
    )

    # --------------------------------------------------
    # Retrievers
    # --------------------------------------------------

    dense = DenseRetriever(
        vector_store,
        embedding_engine,
    )

    keyword = KeywordRetriever(
        bm25
    )

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = (
        "What component controls "
        "the specialized agents?"
    )

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------

    dense_results = dense.search(
        query=query,
        limit=5,
    )

    # --------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------

    keyword_results = keyword.search(
        query=query,
        limit=5,
    )

    # --------------------------------------------------
    # RRF
    # --------------------------------------------------

    hybrid_results = (
        reciprocal_rank_fusion(
            [
                dense_results,
                keyword_results,
            ],
            k=60,
            limit=5,
        )
    )

    print(
        f"\nRRF Results: "
        f"{len(hybrid_results)}"
    )

    # --------------------------------------------------
    # Context Builder
    # --------------------------------------------------

    builder = ContextBuilder(
        max_chunks=5,
        max_context_chars=8000,
        max_chunk_chars=3000,
    )

    context = builder.build(
        query=query,
        results=hybrid_results,
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT BUILDER OUTPUT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSelected Evidence: "
        f"{len(context.items)}"
    )

    print(
        f"Context Characters: "
        f"{len(context.context_text)}"
    )

    print(
        "\n" + "-" * 70
    )

    print(
        context.context_text
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT BUILDER TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()