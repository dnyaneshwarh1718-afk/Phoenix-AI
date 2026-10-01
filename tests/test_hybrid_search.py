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


def main():

    print("=" * 70)
    print("PHOENIX AI - HYBRID RETRIEVAL TEST")
    print("=" * 70)

    # ==============================================================
    # 1. Load document
    # ==============================================================

    file_path = "tests/rag_test_corpus.txt"

    manager = DocumentManager()

    document = manager.load_document(
        file_path
    )

    # ==============================================================
    # 2. Chunk document
    # ==============================================================

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

    # ==============================================================
    # 3. Build BM25 index
    # ==============================================================

    bm25 = BM25Store()

    bm25.build(
        chunks
    )

    # ==============================================================
    # 4. Initialize embedding engine
    # ==============================================================

    embeddings = EmbeddingEngine(
        model_name="nomic-embed-text"
    )

    # ==============================================================
    # 5. Initialize Qdrant
    # ==============================================================

    vector_store = QdrantStore(
        collection_name="phoenix_documents",
        vector_size=768,
    )

    # ==============================================================
    # 6. Create retrievers
    #
    # IMPORTANT:
    # DenseRetriever expects:
    #
    #     DenseRetriever(vector_store, embedding_engine)
    #
    # NOT:
    #
    #     DenseRetriever(embedding_engine, vector_store)
    # ==============================================================

    dense = DenseRetriever(
        vector_store=vector_store,
        embedding_engine=embeddings,
    )

    keyword = KeywordRetriever(
        bm25
    )

    # ==============================================================
    # 7. Query
    # ==============================================================

    query = (
        "What component controls the specialized agents?"
    )

    print(
        f"\nQuery: {query}"
    )

    # ==============================================================
    # 8. Dense retrieval
    # ==============================================================

    dense_results = dense.search(
        query=query,
        limit=5,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DENSE RESULTS"
    )

    print(
        "=" * 70
    )

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: {result.score}"
        )

        print(
            f"Chunk ID: {result.chunk_id}"
        )

        print(
            f"Document ID: {result.document_id}"
        )

        print(
            f"Method: {result.retrieval_method}"
        )

        print(
            f"\n{result.text}"
        )

    # ==============================================================
    # 9. BM25 / Keyword retrieval
    # ==============================================================

    keyword_results = keyword.search(
        query=query,
        limit=5,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BM25 RESULTS"
    )

    print(
        "=" * 70
    )

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: {result.score}"
        )

        print(
            f"Chunk ID: {result.chunk_id}"
        )

        print(
            f"Document ID: {result.document_id}"
        )

        print(
            f"Method: {result.retrieval_method}"
        )

        print(
            f"\n{result.text}"
        )

    # ==============================================================
    # 10. Reciprocal Rank Fusion
    # ==============================================================

    hybrid_results = reciprocal_rank_fusion(
        [
            dense_results,
            keyword_results,
        ],
        k=60,
        limit=5,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID / RRF RESULTS"
    )

    print(
        "=" * 70
    )

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):

        print(
            f"\nRank: {rank}"
        )

        print(
            f"RRF Score: {result.score}"
        )

        print(
            f"Chunk ID: {result.chunk_id}"
        )

        print(
            f"Document ID: {result.document_id}"
        )

        print(
            f"Method: {result.retrieval_method}"
        )

        print(
            f"\n{result.text}"
        )

    # ==============================================================
    # 11. Completion
    # ==============================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID RETRIEVAL TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()