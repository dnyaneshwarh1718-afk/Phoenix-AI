from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.rag.config import RAGConfig
from app.rag.factory import build_rag_engine


def main() -> int:
    settings = Settings()
    cfg = RAGConfig()
    corpus = ROOT / "evaluation" / "corpus"
    supported = {".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv"}

    print("Phoenix corpus index repair")
    print("=" * 70)
    print(f"Project root: {ROOT}")
    print(f"Corpus:       {corpus}")
    print(f"Qdrant:       {cfg.qdrant_host}:{cfg.qdrant_port}/{cfg.collection_name}")
    print(f"BM25:         {Path(cfg.bm25_storage_path).resolve()}")
    print(f"Registry:     {Path(cfg.registry_path).resolve()}")

    engine = build_rag_engine(settings)
    access = engine.document_access_manager
    if access is None:
        print("ERROR: RAG engine has no DocumentAccessManager")
        return 2

    failures = 0
    for path in sorted(corpus.iterdir()):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        try:
            result = access.prepare_document(str(path), auto_index=True)
            if result.ready and result.document:
                record = result.document
                # Never trust the registry alone. Confirm both retrieval
                # backends contain the registered document ID.
                consistent = access.document_indexer.is_index_consistent(
                    Path(record.source_path), record.document_id
                )
                if not consistent:
                    print(f"[REPAIR] {path.name}: registry/backend drift detected; forcing reindex")
                    forced = access.document_indexer.index_document(
                        record.source_path,
                        document_id=record.document_id,
                        force_reindex=True,
                    )
                    if forced.status.name not in {"INDEXED", "REINDEXED", "ALREADY_INDEXED"}:
                        raise RuntimeError(f"forced reindex failed: {forced.message} | {forced.error}")
                    consistent = access.document_indexer.is_index_consistent(
                        Path(record.source_path), record.document_id
                    )
                qcount = access.document_indexer.qdrant_store.document_chunk_count(record.document_id)
                bcount = access.document_indexer.bm25_store.document_chunk_count(record.document_id)
                if not consistent:
                    raise RuntimeError(
                        f"backend validation failed after repair: qdrant={qcount}, bm25={bcount}, expected={record.chunk_count}"
                    )
                print(f"[INDEXED] {path.name}")
                print(f"  path: {record.source_path}")
                print(f"  document_id: {record.document_id}")
                print(f"  registry_chunks: {record.chunk_count}")
                print(f"  qdrant_chunks: {qcount}")
                print(f"  bm25_chunks: {bcount}")
                print(f"  ready: {result.ready}")
            else:
                failures += 1
                print(f"[FAILED]  {path.name}")
                print(f"  status: {result.status}")
                print(f"  message: {result.message}")
        except Exception as exc:
            failures += 1
            print(f"[ERROR]   {path.name}: {type(exc).__name__}: {exc}")

    print("=" * 70)
    if failures:
        print(f"RESULT: FAIL ({failures} corpus files failed)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
