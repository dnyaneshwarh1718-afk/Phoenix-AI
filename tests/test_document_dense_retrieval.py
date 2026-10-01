from app.rag.embeddings.embedding_engine import (
    EmbeddingEngine,
)

from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)

from app.rag.retrieval.dense_retriever import (
    DenseRetriever,
)


def main():

    print("=" * 70)
    print("PHOENIX AI - DOCUMENT DENSE RETRIEVAL TEST")
    print("=" * 70)

    # ==================================================
    # COMPONENTS
    # ==================================================

    embedding_engine = EmbeddingEngine(
        model_name="nomic-embed-text"
    )

    vector_store = QdrantStore(
        collection_name="phoenix_documents"
    )

    retriever = DenseRetriever(
        vector_store=vector_store,
        embedding_engine=embedding_engine,
    )

    # ==================================================
    # TEST DOCUMENT ID
    # ==================================================

    document_id = (
        "5b13cbddc694aa93"
    )

    query = (
        "What is the purpose of this document?"
    )

    print(
        f"\nDocument ID: {document_id}"
    )

    print(
        f"Query: {query}"
    )

    # ==================================================
    # DOCUMENT SEARCH
    # ==================================================

    results = retriever.search_document(
        query=query,
        document_id=document_id,
        limit=5,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DOCUMENT-SPECIFIC DENSE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults: {len(results)}"
    )

    # ==================================================
    # VERIFY RESULTS
    # ==================================================

    passed = True

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n[Result {rank}]"
        )

        print(
            f"Chunk ID: {result.chunk_id}"
        )

        print(
            f"Document ID: {result.document_id}"
        )

        print(
            f"Score: {result.score:.6f}"
        )

        print(
            f"Method: {result.retrieval_method}"
        )

        print(
            f"Text:\n{result.text}"
        )

        # --------------------------------------------------
        # CRITICAL ISOLATION CHECK
        # --------------------------------------------------

        if (
            result.document_id
            != document_id
        ):
            passed = False

            print(
                "\nERROR: Result belongs to "
                "another document!"
            )

    # ==================================================
    # STATUS
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    if passed:
        print(
            "DOCUMENT DENSE RETRIEVAL STATUS: PASSED"
        )
    else:
        print(
            "DOCUMENT DENSE RETRIEVAL STATUS: FAILED"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()