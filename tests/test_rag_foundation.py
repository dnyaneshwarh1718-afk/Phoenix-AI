
from pathlib import Path
import sys

from app.rag.document_manager import DocumentManager
from app.rag.chunking.chunker import DocumentChunker
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.vector_store.qdrant_store import QdrantStore


def main(file_path: str):
    manager = DocumentManager()
    document = manager.load_document(file_path)

    print(f"Document: {document.file_name}")
    print(f"ID: {document.document_id}")
    print(f"Characters: {len(document.text)}")

    chunker = DocumentChunker()
    chunks = chunker.chunk(document)

    print(f"Chunks: {len(chunks)}")

    embeddings = EmbeddingEngine(
        model_name="nomic-embed-text"
    )

    vectors = embeddings.embed_documents(
        [chunk.text for chunk in chunks]
    )

    print(f"Embedding dimension: {len(vectors[0])}")

    store = QdrantStore(
        collection_name="phoenix_documents"
    )

    store.upsert_chunks(chunks, vectors)

    print(f"Qdrant points: {store.count()}")
    print("RAG foundation ingestion: OK")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python tests/test_rag_foundation.py <file>"
        )

    main(sys.argv[1])
