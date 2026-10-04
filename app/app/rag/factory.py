from __future__ import annotations

from app.core.config import Settings
from app.rag.access.document_access_manager import DocumentAccessManager
from app.rag.bm25.bm25_store import BM25Store
from app.rag.chunking.chunker import DocumentChunker
from app.rag.context.context_builder import ContextBuilder
from app.rag.document_manager import DocumentManager
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.generation.answer_generator import AnswerGenerator
from app.rag.generation.prompt_builder import PromptBuilder
from app.rag.indexing.document_indexer import DocumentIndexer
from app.rag.registry.document_registry import DocumentRegistry
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.rag_engine import RAGEngine
from app.rag.validation.answer_validator import AnswerValidator
from app.rag.vector_store.qdrant_store import QdrantStore
from app.rag.discovery.document_discovery import DocumentDiscovery
from app.rag.config import RAGConfig


def build_rag_engine(settings: Settings) -> RAGEngine:
    cfg = RAGConfig()

    embedding = EmbeddingEngine(
        model_name=cfg.embedding_model,
        base_url=cfg.ollama_base_url,
        timeout=cfg.embedding_timeout,
    )
    qdrant = QdrantStore(
        host=cfg.qdrant_host,
        port=cfg.qdrant_port,
        collection_name=cfg.collection_name,
        vector_size=cfg.vector_size,
    )
    bm25 = BM25Store(cfg.bm25_storage_path)
    registry = DocumentRegistry(cfg.registry_path)
    manager = DocumentManager()
    chunker = DocumentChunker(cfg.chunk_size, cfg.chunk_overlap)
    indexer = DocumentIndexer(
        document_manager=manager,
        chunker=chunker,
        embedding_engine=embedding,
        qdrant_store=qdrant,
        bm25_store=bm25,
        registry=registry,
    )
    discovery = DocumentDiscovery()
    access = DocumentAccessManager(
        document_registry=registry,
        document_discovery=discovery,
        document_indexer=indexer,
        default_search_roots=cfg.search_roots,
    )

    retriever = HybridRetriever(
        embedding_engine=embedding,
        vector_store=qdrant,
        bm25_store=bm25,
        rrf_k=cfg.rrf_k,
    )
    context_builder = ContextBuilder(
        max_chunks=cfg.context_max_chunks,
        max_context_chars=cfg.max_context_chars,
        max_chunk_chars=cfg.max_chunk_chars,
    )
    prompt_builder = PromptBuilder()
    generator = AnswerGenerator(
        model_name=settings.ollama_model,
        base_url=settings.ollama_base_url,
        timeout=cfg.generation_timeout,
    )
    validator = AnswerValidator(min_confidence=cfg.min_validation_confidence)

    return RAGEngine(
        hybrid_retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=generator,
        answer_validator=validator,
        document_access_manager=access,
        retrieval_limit=cfg.retrieval_limit,
        candidate_limit=cfg.candidate_limit,
    )
