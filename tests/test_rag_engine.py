from app.rag.document_manager import DocumentManager
from app.rag.chunking.chunker import DocumentChunker

from app.rag.embeddings.embedding_engine import (
    EmbeddingEngine,
)

from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)

from app.rag.bm25.bm25_store import (
    BM25Store,
)

from app.rag.context.context_builder import (
    ContextBuilder,
)

from app.rag.generation.prompt_builder import (
    PromptBuilder,
)

from app.rag.generation.answer_generator import (
    AnswerGenerator,
)

from app.rag.validation.answer_validator import (
    AnswerValidator,
)

from app.rag.rag_engine import (
    RAGEngine,
)


def main():

    print("=" * 70)
    print("PHOENIX AI - END-TO-END RAG TEST")
    print("=" * 70)

    # ==================================================
    # DOCUMENT
    # ==================================================

    file_path = "tests/rag_test_corpus.txt"

    manager = DocumentManager()

    document = manager.load_document(
        file_path
    )

    print(
        f"\nDocument: {document.file_name}"
    )

    # ==================================================
    # CHUNKING
    # ==================================================

    chunker = DocumentChunker()

    chunks = chunker.chunk(
        document
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    # ==================================================
    # BM25 INDEX
    # ==================================================

    bm25_store = BM25Store()

    bm25_store.build(
        chunks
    )

    # ==================================================
    # EMBEDDING ENGINE
    # ==================================================

    embedding_engine = EmbeddingEngine(
        model_name="nomic-embed-text"
    )

    # ==================================================
    # VECTOR STORE
    # ==================================================

    vector_store = QdrantStore(
        collection_name="phoenix_documents"
    )

    # ==================================================
    # CONTEXT BUILDER
    # ==================================================

    context_builder = ContextBuilder(
        max_chunks=5,
        max_context_chars=8000,
        max_chunk_chars=3000,
    )

    # ==================================================
    # PROMPT BUILDER
    # ==================================================

    prompt_builder = PromptBuilder()

    # ==================================================
    # ANSWER GENERATOR
    # ==================================================

    answer_generator = AnswerGenerator(
        model_name="qwen3:4b-instruct"
    )

    # ==================================================
    # ANSWER VALIDATOR
    # ==================================================

    answer_validator = AnswerValidator(
        min_confidence=0.50
    )

    # ==================================================
    # RAG ENGINE
    # ==================================================

    rag_engine = RAGEngine(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        bm25_store=bm25_store,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        answer_generator=answer_generator,
        answer_validator=answer_validator,
        retrieval_limit=5,
        rrf_k=60,
    )

    # ==================================================
    # QUERY
    # ==================================================

    query = (
        "What component controls the "
        "specialized agents?"
    )

    print(
        f"\nQuery: {query}"
    )

    # ==================================================
    # EXECUTE RAG
    # ==================================================

    result = rag_engine.ask(
        query
    )

    # ==================================================
    # FINAL ANSWER
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RAG ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAnswer:\n{result.answer}"
    )

    # ==================================================
    # MODEL
    # ==================================================

    print(
        f"\nModel: {result.model}"
    )

    # ==================================================
    # EVIDENCE
    # ==================================================

    print(
        f"Evidence Count: "
        f"{result.evidence_count}"
    )

    print(
        f"Context Characters: "
        f"{result.context_characters}"
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ANSWER VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nValidation Valid: "
        f"{result.valid}"
    )

    print(
        f"Validation Confidence: "
        f"{result.confidence:.4f}"
    )

    print(
        f"Validation Reason: "
        f"{result.validation_reason}"
    )

    # ==================================================
    # RETRIEVED EVIDENCE
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVED EVIDENCE"
    )

    print(
        "=" * 70
    )

    for rank, evidence in enumerate(
        result.evidence,
        start=1,
    ):

        print(
            f"\n[Evidence {rank}]"
        )

        print(
            f"Chunk ID: "
            f"{evidence.chunk_id}"
        )

        print(
            f"Document ID: "
            f"{evidence.document_id}"
        )

        print(
            f"Score: "
            f"{evidence.score:.6f}"
        )

        print(
            f"Method: "
            f"{evidence.retrieval_method}"
        )

        print(
            f"Metadata: "
            f"{evidence.metadata}"
        )

    # ==================================================
    # TEST STATUS
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    if result.valid:
        print(
            "RAG TEST STATUS: PASSED"
        )
    else:
        print(
            "RAG TEST STATUS: FAILED"
        )

    print(
        "=" * 70
    )

    print(
        "\nEND-TO-END RAG TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()