from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Always make the Phoenix project root importable when this script is invoked as
# `python evaluation/scripts/diagnose_document_retrieval.py ...` from the root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.rag.config import RAGConfig
from app.rag.document_resolution.document_resolver import DocumentResolver
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.registry.document_registry import DocumentRegistry
from app.rag.vector_store.qdrant_store import QdrantStore
from app.rag.bm25.bm25_store import BM25Store


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phoenix document-scoped RAG retrieval")
    parser.add_argument("document", help="Document path")
    parser.add_argument("query", help="Question to test")
    args = parser.parse_args()

    document = Path(args.document).resolve()
    print("Phoenix document retrieval diagnostic")
    print("=" * 70)
    print(f"Project root: {ROOT}")
    print(f"Document:     {document}")
    print(f"Exists:       {document.exists()}")
    print(f"Query:        {args.query}")

    if not document.exists() or not document.is_file():
        print("ERROR: document does not exist or is not a file")
        return 2

    cfg = RAGConfig()
    settings = Settings()
    # The resolver requires the same persistent registry used by the RAG
    # factory. The previous diagnostic instantiated DocumentResolver() with no
    # registry, which caused the misleading constructor error and prevented
    # retrieval diagnostics from ever reaching Qdrant/BM25.
    registry = DocumentRegistry(cfg.registry_path)
    resolver = DocumentResolver(document_registry=registry)

    resolved = resolver.resolve(str(document))
    if not resolved.found or resolved.document is None:
        # Also try the exact filename so diagnostics remain useful when the
        # registry stores a normalized path from a different Windows prefix.
        resolved = resolver.resolve(document.name)

    print(f"Resolver found: {resolved.found}")
    print(f"Resolver match: {resolved.match_type}")
    print(f"Resolver msg:   {resolved.message}")

    if not resolved.found or resolved.document is None:
        print("ERROR: no registered document could be resolved")
        print(f"Registry path: {Path(cfg.registry_path).resolve()}")
        return 3

    record = resolved.document
    document_id = record.document_id
    print(f"Document ID:    {document_id}")
    print(f"Registered path:{record.source_path}")
    print(f"Chunk count:    {record.chunk_count}")
    print(f"Registry status:{record.status}")

    qdrant = QdrantStore(
        host=cfg.qdrant_host,
        port=cfg.qdrant_port,
        collection_name=cfg.collection_name,
        vector_size=cfg.vector_size,
    )
    bm25 = BM25Store(cfg.bm25_storage_path)
    embedder = EmbeddingEngine(
        model_name=cfg.embedding_model,
        base_url=cfg.ollama_base_url,
        timeout=cfg.embedding_timeout,
    )
    vector = embedder.embed_query(args.query)

    print(f"Embedding model: {cfg.embedding_model}")
    print(f"Embedding dimension: {len(vector)}")
    print(f"Qdrant total chunks:  {qdrant.count()}")
    print(f"Qdrant doc chunks:    {qdrant.document_chunk_count(document_id)}")
    print(f"BM25 total chunks:    {bm25.count()}")
    print(f"BM25 doc chunks:      {bm25.document_chunk_count(document_id)}")

    dense = qdrant.search_document(vector, document_id, limit=5)
    lexical = bm25.search_document(args.query, document_id, limit=5)

    print(f"Scoped dense results: {len(dense)}")
    for i, point in enumerate(dense, 1):
        payload = point.payload or {}
        text = str(payload.get("text", ""))
        print(f"  D{i}: score={getattr(point, 'score', None):.4f} chunk={payload.get('chunk_id')}")
        print(f"      {text[:300].replace(chr(10), ' ')}")

    print(f"Scoped BM25 results:  {len(lexical)}")
    for i, item in enumerate(lexical, 1):
        print(f"  B{i}: score={item.score:.4f} chunk={item.chunk.chunk_id}")
        print(f"      {item.chunk.text[:300].replace(chr(10), ' ')}")

    global_points = qdrant.search(vector, limit=20)
    global_doc_matches = [
        p for p in global_points
        if str((p.payload or {}).get("document_id", "")) == str(document_id)
    ]
    print(f"Global dense results: {len(global_points)}")
    print(f"Global matching doc:  {len(global_doc_matches)}")

    result = {
        "document_id": document_id,
        "document_path": record.source_path,
        "qdrant_document_chunks": qdrant.document_chunk_count(document_id),
        "bm25_document_chunks": bm25.document_chunk_count(document_id),
        "scoped_dense_results": len(dense),
        "scoped_bm25_results": len(lexical),
        "global_document_matches": len(global_doc_matches),
        "embedding_model": cfg.embedding_model,
        "embedding_dimension": len(vector),
    }
    print("\nSUMMARY")
    print(json.dumps(result, indent=2))

    return 0 if (dense or lexical) else 4


if __name__ == "__main__":
    raise SystemExit(main())
