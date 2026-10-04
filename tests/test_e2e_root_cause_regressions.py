from pathlib import Path
from types import SimpleNamespace

from app.orchestrator.router import IntentRouter
from app.rag.access.document_access_manager import DocumentAccessManager


def test_memory_write_wins_over_project_knowledge_terms():
    assert IntentRouter().classify(
        "Remember that my primary Phoenix AI vector database is Qdrant."
    ) == "memory"


def test_project_knowledge_question_routes_to_rag():
    assert IntentRouter().classify("What vector database does Phoenix AI use?") == "rag"


def test_document_backend_consistency_is_required():
    manager = object.__new__(DocumentAccessManager)
    record = SimpleNamespace(document_id="doc-1", chunk_count=3)
    manager.document_indexer = SimpleNamespace(
        qdrant_store=SimpleNamespace(document_chunk_count=lambda _: 2),
        bm25_store=SimpleNamespace(document_chunk_count=lambda _: 3),
    )
    assert manager._backends_consistent(record) is False

    manager.document_indexer.qdrant_store.document_chunk_count = lambda _: 3
    assert manager._backends_consistent(record) is True
