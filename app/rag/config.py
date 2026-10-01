from __future__ import annotations

from dataclasses import dataclass, field
import os


def _roots() -> list[str]:
    raw = os.getenv("RAG_SEARCH_ROOTS", "R:\\")
    return [item.strip() for item in raw.split(";") if item.strip()]


@dataclass(frozen=True)
class RAGConfig:
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "phoenix_documents")
    vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embedding_timeout: float = float(os.getenv("RAG_EMBEDDING_TIMEOUT", "120"))
    generation_timeout: int = int(os.getenv("RAG_GENERATION_TIMEOUT", "180"))

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    retrieval_limit: int = int(os.getenv("RAG_RETRIEVAL_LIMIT", "5"))
    candidate_limit: int = int(os.getenv("RAG_CANDIDATE_LIMIT", "20"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))

    context_max_chunks: int = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "5"))
    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "9000"))
    max_chunk_chars: int = int(os.getenv("RAG_MAX_CHUNK_CHARS", "3000"))
    min_validation_confidence: float = float(os.getenv("RAG_MIN_VALIDATION_CONFIDENCE", "0.35"))

    bm25_storage_path: str = os.getenv("RAG_BM25_STORAGE", "data/bm25_chunks.json")
    registry_path: str = os.getenv("RAG_REGISTRY_PATH", "data/document_registry.json")
    search_roots: list[str] = field(default_factory=_roots)
