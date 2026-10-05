"""Force-reindex Phoenix's evaluation corpus with the current ingestion pipeline.

Use this after changing a loader/chunker so an existing persistent registry does
not keep serving stale chunks during evaluation.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.rag.factory import build_rag_engine


FILES = (
    "phoenix_evaluation.xlsx",
    "phoenix_evaluation.docx",
    "phoenix_evaluation.pdf",
    "phoenix_evaluation.pptx",
)


def main() -> int:
    engine = build_rag_engine(get_settings())
    access = engine.document_access_manager
    roots = access.default_search_roots
    corpus = Path(__file__).resolve().parents[1] / "corpus"

    failures = 0
    for name in FILES:
        path = corpus / name
        result = access.document_indexer.index_document(str(path), force_reindex=True)
        print(f"[{name}] {result.status.value}: chunks={result.chunk_count} message={result.message}")
        if result.status.value not in {"INDEXED", "REINDEXED", "ALREADY_INDEXED"}:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
