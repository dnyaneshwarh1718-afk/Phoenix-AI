import pytest

from app.agents.base_agent import AgentContext
from app.agents.rag_agent import RAGAgent
from app.rag.rag_engine import RAGResponse


class FakeEngine:
    def ask(self, query, document_id=None):
        return RAGResponse(query, "grounded answer", "test-model", 2, 100, True, .9, "ok", [], [{"citation_id": 1}], document_id, None, "ok")

    def ask_document(self, query, document_reference, search_roots=None, *, auto_index=True):
        return RAGResponse(query, "document answer", "test-model", 1, 50, True, .8, "ok", [], [{"citation_id": 1}], "doc-1", document_reference, "ok")


@pytest.mark.asyncio
async def test_rag_agent_global_query():
    result = await RAGAgent(FakeEngine()).run("What is SQL?", AgentContext("req"))
    assert result.success is True
    assert result.content == "grounded answer"
    assert result.data["citations"]


@pytest.mark.asyncio
async def test_rag_agent_document_query():
    context = AgentContext("req", metadata={"document_reference": "SQL INTERVIEW QUESTIONS.docx"})
    result = await RAGAgent(FakeEngine()).run("What is SQL?", context)
    assert result.success is True
    assert result.data["document_id"] == "doc-1"
