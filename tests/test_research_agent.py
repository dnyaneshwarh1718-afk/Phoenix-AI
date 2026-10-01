import pytest

from app.agents.base_agent import AgentContext
from app.agents.research_agent import ResearchAgent
from app.research.models import ResearchSource


class FakeSearch:
    name = "fake-search"

    async def search(self, request):
        assert request.query == "latest RAG techniques"
        return [
            ResearchSource("Source One", "https://example.com/one", "RAG combines retrieval and generation.", "example.com", 1),
            ResearchSource("Source Two", "https://example.com/two", "Hybrid retrieval improves lexical and semantic coverage.", "example.com", 2),
        ]


class FakeLLMResponse:
    content = "RAG combines retrieval and generation [S1]. Hybrid retrieval improves coverage [S2].\n\nSources\n[S1] https://example.com/one\n[S2] https://example.com/two"
    provider = "ollama"
    model = "qwen3:4b-instruct"


class FakeLLM:
    async def chat(self, request):
        assert request.task_type == "research"
        assert "[S1]" in request.messages[1]["content"]
        return FakeLLMResponse()


@pytest.mark.asyncio
async def test_research_agent_searches_and_cites_sources():
    result = await ResearchAgent(FakeLLM(), FakeSearch()).run(
        "latest RAG techniques", AgentContext("req")
    )
    assert result.success
    assert result.data["status"] == "ok"
    assert result.data["source_count"] == 2
    assert "[S1]" in result.content
    assert result.data["sources"][0]["url"] == "https://example.com/one"


class EmptySearch:
    name = "empty"
    async def search(self, request):
        return []


@pytest.mark.asyncio
async def test_research_agent_handles_no_sources_without_llm():
    class ShouldNotRun:
        async def chat(self, request):
            raise AssertionError("LLM must not run when no sources exist")
    result = await ResearchAgent(ShouldNotRun(), EmptySearch()).run("unknown", AgentContext("req"))
    assert result.success
    assert result.data["status"] == "no_sources"


class FailingSearch:
    name = "failing"
    async def search(self, request):
        raise RuntimeError("network down")


@pytest.mark.asyncio
async def test_research_agent_handles_search_failure():
    result = await ResearchAgent(FakeLLM(), FailingSearch()).run("latest RAG techniques", AgentContext("req"))
    assert not result.success
    assert result.data["status"] == "search_failed"


class UncitedLLM:
    async def chat(self, request):
        return FakeLLMResponseNoCitations()


class FakeLLMResponseNoCitations:
    content = "This answer contains no source references."
    provider = "ollama"
    model = "qwen3:4b-instruct"


@pytest.mark.asyncio
async def test_research_agent_rejects_uncited_synthesis():
    result = await ResearchAgent(UncitedLLM(), FakeSearch()).run("latest RAG techniques", AgentContext("req"))
    assert not result.success
    assert result.data["status"] == "unverified"
    assert result.data["source_count"] == 2
