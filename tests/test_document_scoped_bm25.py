from app.rag.bm25.bm25_store import BM25Store
from app.rag.models import DocumentChunk


def chunk(cid, did, text, idx):
    return DocumentChunk(
        chunk_id=cid,
        document_id=did,
        text=text,
        chunk_index=idx,
        metadata={},
    )


def main():
    print("=" * 70)
    print("PHOENIX AI - DOCUMENT-SCOPED BM25 TEST")
    print("=" * 70)

    store = BM25Store()
    store.add_chunks([
        chunk("a1", "doc-a", "Python pandas data analysis", 0),
        chunk("a2", "doc-a", "Power BI dashboard analysis", 1),
        chunk("b1", "doc-b", "Qdrant vector database retrieval", 0),
        chunk("b2", "doc-b", "BM25 keyword retrieval", 1),
    ])

    results = store.search_document(
        "vector database",
        document_id="doc-b",
        limit=5,
    )

    assert results
    assert all(result.document_id == "doc-b" for result in results)

    global_results = store.search("retrieval", limit=5)
    assert global_results

    print(f"Document-specific results: {len(results)}")
    print("Isolation check: PASS")
    print("Global search compatibility: PASS")

    print("\n" + "=" * 70)
    print("DOCUMENT-SCOPED BM25 TEST STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
