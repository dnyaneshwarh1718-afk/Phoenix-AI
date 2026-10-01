from pathlib import Path

from app.rag.document_manager import DocumentManager
from app.rag.chunking.chunker import DocumentChunker
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.vector_store.qdrant_store import QdrantStore
from app.rag.bm25.bm25_store import BM25Store
from app.rag.discovery.document_registry import DocumentRegistry
from app.rag.indexing.document_indexer import (
    DocumentIndexer,
    IndexStatus,
)


REAL_DOCUMENT = Path(
    r"R:\SQL INTERVIEW QUESTIONS.docx"
)


def test_real_document_indexing():

    print("\n")
    print("=" * 70)
    print("PHOENIX AI - REAL DOCUMENT INDEXING TEST")
    print("=" * 70)

    print(f"\nDocument:")
    print(f"  {REAL_DOCUMENT}")

    assert REAL_DOCUMENT.exists(), (
        f"Test document does not exist: {REAL_DOCUMENT}"
    )

    # --------------------------------------------------
    # Create REAL Phoenix components
    # --------------------------------------------------

    document_manager = DocumentManager()

    chunker = DocumentChunker(
        chunk_size=1000,
        chunk_overlap=150,
    )

    embedding_engine = EmbeddingEngine()

    qdrant_store = QdrantStore()

    bm25_store = BM25Store()

    registry = DocumentRegistry()

    # --------------------------------------------------
    # Create REAL indexer
    # --------------------------------------------------

    indexer = DocumentIndexer(
        document_manager=document_manager,
        chunker=chunker,
        embedding_engine=embedding_engine,
        qdrant_store=qdrant_store,
        bm25_store=bm25_store,
        registry=registry,
    )

    # --------------------------------------------------
    # Index the real document
    # --------------------------------------------------

    result = indexer.index(
        REAL_DOCUMENT
    )

    # --------------------------------------------------
    # Print result
    # --------------------------------------------------

    print("\nResult")
    print("-" * 70)

    print(f"Status:       {result.status}")
    print(f"Document ID:  {result.document_id}")
    print(f"Chunks:       {result.chunk_count}")
    print(f"Path:         {result.path}")
    print(f"Message:      {result.message}")

    if result.error:
        print(f"Error:        {result.error}")

    print("=" * 70)

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert result.status in {
        IndexStatus.INDEXED,
        IndexStatus.ALREADY_INDEXED,
    }

    assert result.document_id is not None

    if result.status == IndexStatus.INDEXED:
        assert result.chunk_count > 0
        