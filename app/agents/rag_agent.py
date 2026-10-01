from __future__ import annotations

import asyncio
from typing import Any

from app.agents.base_agent import AgentContext, AgentResult, BaseAgent
from app.rag.rag_engine import RAGEngine, RAGResponse


class RAGAgent(BaseAgent):
    """Phoenix grounded knowledge agent backed by the complete local RAG pipeline."""

    name = "rag"

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        if not message or not message.strip():
            return AgentResult(success=False, content="", error="RAG query cannot be empty.")

        metadata = context.metadata or {}
        document_reference = metadata.get("document_reference")
        search_roots = metadata.get("search_roots")
        auto_index = bool(metadata.get("auto_index", True))

        try:
            if document_reference:
                response = await asyncio.to_thread(
                    self.rag_engine.ask_document,
                    message,
                    str(document_reference),
                    search_roots,
                    auto_index=auto_index,
                )
            else:
                response = await asyncio.to_thread(
                    self.rag_engine.ask,
                    message,
                )
        except Exception as exc:
            return AgentResult(
                success=False,
                content="",
                error=f"RAG execution failed: {type(exc).__name__}: {exc}",
            )

        return self._to_agent_result(response)

    @staticmethod
    def _to_agent_result(response: RAGResponse) -> AgentResult:
        return AgentResult(
            success=response.valid or response.status in {"no_retrieval", "no_context", "not_found", "ambiguous", "indexed", "discovered_not_indexed"},
            content=response.answer,
            data={
                "status": response.status,
                "valid": response.valid,
                "confidence": response.confidence,
                "validation_reason": response.validation_reason,
                "model": response.model,
                "evidence_count": response.evidence_count,
                "context_characters": response.context_characters,
                "document_id": response.document_id,
                "source_path": response.source_path,
                "citations": response.citations,
            },
            error=None if response.status != "unverified" else response.validation_reason,
        )
