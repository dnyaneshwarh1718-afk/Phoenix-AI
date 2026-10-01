from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.rag.access.document_access_manager import DocumentAccessManager
from app.rag.citations.citation_engine import CitationEngine
from app.rag.context.context_builder import ContextBuilder, RetrievalContext
from app.rag.generation.answer_generator import AnswerGenerator, AnswerGenerationResult
from app.rag.generation.prompt_builder import PromptBuilder
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.validation.answer_validator import AnswerValidationResult, AnswerValidator


@dataclass
class RAGResponse:
    query: str
    answer: str
    model: str
    evidence_count: int
    context_characters: int
    valid: bool
    confidence: float
    validation_reason: str
    evidence: list[Any] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    document_id: str | None = None
    source_path: str | None = None
    status: str = "ok"


class RAGEngine:
    """Production RAG application service: retrieve -> rerank -> ground -> generate -> validate -> cite."""

    def __init__(
        self,
        *args,
        hybrid_retriever: HybridRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        answer_generator: AnswerGenerator | None = None,
        answer_validator: AnswerValidator | None = None,
        document_access_manager: DocumentAccessManager | None = None,
        retrieval_limit: int = 5,
        candidate_limit: int = 20,
        citation_engine: CitationEngine | None = None,
        embedding_engine=None,
        vector_store=None,
        bm25_store=None,
        rrf_k: int = 60,
    ) -> None:
        # Backward-compatible constructor: the previous Phoenix API accepted
        # embedding_engine, vector_store, bm25_store, context_builder,
        # prompt_builder, answer_generator, answer_validator positionally.
        if args:
            if isinstance(args[0], HybridRetriever):
                hybrid_retriever = args[0]
                if len(args) > 1: context_builder = args[1]
                if len(args) > 2: prompt_builder = args[2]
                if len(args) > 3: answer_generator = args[3]
                if len(args) > 4: answer_validator = args[4]
                if len(args) > 5: document_access_manager = args[5]
            else:
                names = ["embedding_engine", "vector_store", "bm25_store", "context_builder", "prompt_builder", "answer_generator", "answer_validator"]
                values = list(args[:len(names)])
                for name, value in zip(names, values):
                    if name == "embedding_engine": embedding_engine = value
                    elif name == "vector_store": vector_store = value
                    elif name == "bm25_store": bm25_store = value
                    elif name == "context_builder": context_builder = value
                    elif name == "prompt_builder": prompt_builder = value
                    elif name == "answer_generator": answer_generator = value
                    elif name == "answer_validator": answer_validator = value

        if hybrid_retriever is None:
            if embedding_engine is None or vector_store is None or bm25_store is None:
                raise ValueError("RAGEngine requires a HybridRetriever or embedding/vector/BM25 dependencies.")
            hybrid_retriever = HybridRetriever(
                embedding_engine=embedding_engine,
                vector_store=vector_store,
                bm25_store=bm25_store,
                rrf_k=rrf_k,
            )
        if context_builder is None or prompt_builder is None or answer_generator is None:
            raise ValueError("RAGEngine requires context_builder, prompt_builder, and answer_generator.")
        if answer_validator is None:
            answer_validator = AnswerValidator(min_confidence=0.35)
        if retrieval_limit <= 0 or candidate_limit <= 0:
            raise ValueError("Retrieval limits must be positive.")

        self.hybrid_retriever = hybrid_retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.answer_generator = answer_generator
        self.answer_validator = answer_validator
        self.document_access_manager = document_access_manager
        self.retrieval_limit = retrieval_limit
        self.candidate_limit = max(candidate_limit, retrieval_limit)
        self.citation_engine = citation_engine or CitationEngine()

    def retrieve(self, query: str, document_id: str | None = None):
        if not query or not query.strip():
            return []
        return self.hybrid_retriever.search(
            query.strip(),
            limit=self.retrieval_limit,
            candidate_limit=self.candidate_limit,
            document_id=document_id,
        )

    def build_context(self, query: str, results) -> RetrievalContext:
        return self.context_builder.build(query=query, results=results)

    def generate_answer(self, query: str, context: str) -> AnswerGenerationResult:
        return self.answer_generator.generate_from_context(
            query=query,
            context=context,
            prompt_builder=self.prompt_builder,
        )

    def validate_answer(self, answer: str, context_result: RetrievalContext) -> AnswerValidationResult:
        return self.answer_validator.validate(answer=answer, context=context_result)

    def ask(self, query: str, document_id: str | None = None) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        query = query.strip()
        results = self.retrieve(query, document_id)

        if not results:
            return self._empty_response(
                query,
                "No relevant indexed evidence was found.",
                document_id=document_id,
                status="no_retrieval",
            )

        context_result = self.build_context(query, results)
        if not context_result.items or not context_result.context_text:
            return self._empty_response(
                query,
                "Relevant chunks were found, but sufficient context could not be constructed.",
                document_id=document_id,
                status="no_context",
            )

        try:
            generation = self.generate_answer(query, context_result.context_text)
        except Exception:
            return self._empty_response(
                query,
                "The answer generator is currently unavailable. Please verify that Ollama is running and try again.",
                document_id=document_id,
                status="generation_failed",
            )

        validation = self.validate_answer(generation.answer, context_result)
        citations = self.citation_engine.build(context_result.items)

        if validation.is_valid:
            answer = generation.answer.strip()
            status = "ok"
        else:
            answer = "I could not verify the generated answer against the retrieved documents."
            status = "unverified"

        return RAGResponse(
            query=query,
            answer=answer,
            model=generation.model,
            evidence_count=len(context_result.items),
            context_characters=len(context_result.context_text),
            valid=validation.is_valid,
            confidence=float(validation.confidence),
            validation_reason=validation.reason,
            evidence=context_result.items,
            citations=citations,
            document_id=document_id,
            status=status,
        )

    def ask_document(
        self,
        query: str,
        document_reference: str,
        search_roots: list[str] | None = None,
        *,
        auto_index: bool = True,
    ) -> RAGResponse:
        if self.document_access_manager is None:
            raise RuntimeError("DocumentAccessManager is required for document-scoped RAG.")
        access = self.document_access_manager.prepare_document(
            document_reference,
            search_roots,
            auto_index=auto_index,
        )
        if not access.ready or access.document is None:
            return self._empty_response(
                query.strip() if query else "",
                access.message,
                document_id=access.document.document_id if access.document else None,
                source_path=access.matched_path,
                status=access.status,
            )
        response = self.ask(query, document_id=access.document.document_id)
        response.source_path = access.document.source_path
        response.document_id = access.document.document_id
        return response

    def _empty_response(self, query: str, message: str, *, document_id=None, source_path=None, status="failed") -> RAGResponse:
        return RAGResponse(
            query=query,
            answer=message,
            model=self.answer_generator.model_name,
            evidence_count=0,
            context_characters=0,
            valid=False,
            confidence=0.0,
            validation_reason=message,
            evidence=[],
            citations=[],
            document_id=document_id,
            source_path=source_path,
            status=status,
        )
