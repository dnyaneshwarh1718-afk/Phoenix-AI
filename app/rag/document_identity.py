from __future__ import annotations

import hashlib
from pathlib import Path


def canonical_document_id(path: str | Path) -> str:
    """Return Phoenix's single canonical, cross-component document ID.

    All ingestion loaders, the indexer, registry and document resolver must
    use exactly this normalization.  The previous implementation had loaders
    hashing the path as-is while DocumentIndexer lower-cased it, which could
    produce different IDs on Windows when drive/path casing differed.
    """
    normalized = str(Path(path).resolve()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
