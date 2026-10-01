from app.rag.document_manager import DocumentManager
from app.rag.chunking.chunker import DocumentChunker
from app.rag.bm25.bm25_store import BM25Store


def main():

    file_path = "tests/rag_test_corpus.txt"

    print("=" * 70)
    print("PHOENIX AI - BM25 RETRIEVAL TEST")
    print("=" * 70)

    manager = DocumentManager()

    document = manager.load_document(file_path)

    print(f"\nDocument: {document.file_name}")

    chunker = DocumentChunker()

    chunks = chunker.chunk(document)

    print(f"Chunks: {len(chunks)}")

    bm25 = BM25Store()

    bm25.build(chunks)

    query = "Orchestrator Agent"

    print(f"\nQuery: {query}")

    results = bm25.search(
        query,
        limit=5,
    )

    print(f"Results: {len(results)}")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(f"Rank: {rank}")
        print(f"Score: {result.score}")
        print(f"Chunk ID: {result.chunk_id}")
        print()
        print(result.text)

        print("\n" + "=" * 70)
        print("BM25 TEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()