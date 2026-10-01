"""
Integration test for the new local-document -> index -> document-scoped RAG path.

Run this from the Phoenix project root after Qdrant and Ollama are available:

    python -m tests.test_document_scoped_rag

The test creates a temporary text document, asks DocumentAccessManager to find
and index it, then asks RAGEngine to answer only from that document.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.access.document_access_manager import DocumentAccessManager
from app.rag.bm25.bm25_store import BM25Store
from app.rag.chunking.chunker import DocumentChunker
from app.rag.context.context_builder import ContextBuilder
from app.rag.document_manager import DocumentManager
from app.rag.discovery.document_discovery import DocumentDiscovery
from app.rag.embeddings.embedding_engine import EmbeddingEngine
from app.rag.generation.answer_generator import AnswerGenerator
from app.rag.generation.prompt_builder import PromptBuilder
from app.rag.indexing.document_indexer import DocumentIndexer
from app.rag.rag_engine import RAGEngine
from app.rag.registry.document_registry import DocumentRegistry
from app.rag.validation.answer_validator import AnswerValidator
from app.rag.vector_store.qdrant_store import QdrantStore


def main():
    print("=" * 70)
    print("PHOENIX AI - DOCUMENT-SCOPED RAG TEST")
    print("=" * 70)

    with TemporaryDirectory() as temp:
        root = Path(temp) / "local_drive"
        root.mkdir()
        target = root / "phoenix_document_test.txt"
        target.write_text(
            """
            PHOENIX TEST DOCUMENT

            Phoenix uses the Orchestrator Agent as the central controller.
            The Orchestrator delegates work to specialized agents.
            The RAG Agent handles document retrieval and context construction.
            """.strip(),
            encoding="utf-8",
        )

        registry = DocumentRegistry(str(Path(temp) / "registry.json"))
        discovery = DocumentDiscovery()
        manager = DocumentManager()
        chunker = DocumentChunker()
        embeddings = EmbeddingEngine(model_name="nomic-embed-text")
        qdrant = QdrantStore(collection_name="phoenix_documents")
        bm25 = BM25Store()
        indexer = DocumentIndexer(
            document_manager=manager,
            chunker=chunker,
            embedding_engine=embeddings,
            vector_store=qdrant,
            bm25_store=bm25,
            document_registry=registry,
        )
        access = DocumentAccessManager(
            document_registry=registry,
            document_discovery=discovery,
            document_indexer=indexer,
            default_search_roots=[str(root)],
        )

        rag = RAGEngine(
            embedding_engine=embeddings,
            vector_store=qdrant,
            bm25_store=bm25,
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            answer_generator=AnswerGenerator(model_name="qwen3:4b-instruct"),
            answer_validator=AnswerValidator(min_confidence=0.50),
            document_access_manager=access,
        )

        query = "What component controls the specialized agents?"
        result = rag.ask_document(
            query=query,
            document_reference="phoenix_document_test",
            search_roots=[str(root)],
        )

        assert result.document_id is not None
        assert result.source_path == str(target.resolve())
        assert result.evidence_count > 0
        assert result.valid
        assert "Orchestrator" in result.answer

        print(f"Document: {result.source_path}")
        print(f"Document ID: {result.document_id}")
        print(f"Evidence Count: {result.evidence_count}")
        print(f"Validation: {result.valid}")
        print(f"Answer: {result.answer}")

    print("\n" + "=" * 70)
    print("DOCUMENT-SCOPED RAG TEST STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
