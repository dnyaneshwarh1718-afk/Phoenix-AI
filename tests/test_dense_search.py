from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.vector_store.qdrant_store import QdrantStore


def main():

    query = "What is Phoenix AI?"

    print("=" * 70)
    print("PHOENIX AI - DENSE RETRIEVAL TEST")
    print("=" * 70)

    embeddings = EmbeddingEngine(
        model_name="nomic-embed-text"
    )

    vector_store = QdrantStore(
        collection_name="phoenix_documents"
    )

    print(f"\nQuery: {query}")

    query_vector = embeddings.embed_query(query)

    print(
        f"Query embedding dimension: "
        f"{len(query_vector)}"
    )

    results = vector_store.search(
        query_vector=query_vector,
        limit=5,
    )

    print(f"\nResults: {len(results)}")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(f"Rank: {rank}")
        print(f"Score: {result.score}")
        print(f"Chunk ID: {result.id}")

        payload = result.payload or {}

        print(
            f"Document: "
            f"{payload.get('file_name')}"
        )

        print(
            f"\nContent:\n"
            f"{payload.get('text', '')}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()