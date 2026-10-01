from app.rag.document_manager import DocumentManager
from app.rag.chunking.chunker import DocumentChunker


def main():

    manager = DocumentManager()

    document = manager.load_document(
        "tests/rag_test_corpus.txt"
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk(
        document
    )

    print("=" * 70)
    print("PHOENIX AI - CHUNK IDENTITY TEST")
    print("=" * 70)

    print(
        f"\nDocument ID: "
        f"{document.document_id}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    seen = set()

    for chunk in chunks:

        print("\n" + "-" * 70)

        print(
            f"Chunk index: "
            f"{chunk.chunk_index}"
        )

        print(
            f"Chunk ID: "
            f"{chunk.chunk_id}"
        )

        print(
            f"Document ID: "
            f"{chunk.document_id}"
        )

        print(
            f"Characters: "
            f"{len(chunk.text)}"
        )

        if chunk.chunk_id in seen:

            print(
                "❌ DUPLICATE CHUNK ID"
            )

        else:

            print(
                "✅ UNIQUE CHUNK ID"
            )

            seen.add(
                chunk.chunk_id
            )

    print("\n" + "=" * 70)

    if len(seen) == len(chunks):

        print(
            "CHUNK IDENTITY TEST: PASS"
        )

    else:

        print(
            "CHUNK IDENTITY TEST: FAIL"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()