import pytest

from app.agents.base_agent import AgentContext
from app.agents.planning_agent import PlanningAgent


class Response:
    provider = "ollama"
    model = "qwen3:4b-instruct"
    content = '''```json
{"goal":"Build a RAG pipeline","steps":[{"id":"step_1","title":"Inspect requirements","description":"Review the available requirements and constraints.","agent":"general","depends_on":[],"tools":[],"approval_required":false},{"id":"step_2","title":"Build retrieval","description":"Implement document retrieval and validation.","agent":"rag","depends_on":["step_1"],"tools":["qdrant"],"approval_required":false}],"blockers":[],"assumptions":[],"notes":[]}
```'''


class FakeLLM:
    async def chat(self, request):
        assert request.task_type == "reasoning"
        assert request.temperature == 0.1
        return Response()


@pytest.mark.asyncio
async def test_planning_agent_returns_valid_structured_plan():
    result = await PlanningAgent(FakeLLM()).run("Build a RAG pipeline", AgentContext("req"))
    assert result.success
    assert result.data["status"] == "ready"
    assert result.data["step_count"] == 2
    assert result.data["plan"]["steps"][1]["depends_on"] == ["step_1"]
    assert "Execution plan:" in result.content


class InvalidLLM:
    async def chat(self, request):
        return type("R", (), {"content": '{"goal":"x","steps":[{"id":"step_1","title":"A","description":"A","depends_on":["step_2"]},{"id":"step_2","title":"B","description":"B","depends_on":["step_1"]}]}', "provider":"ollama", "model":"qwen3:4b-instruct"})()


@pytest.mark.asyncio
async def test_planning_agent_rejects_cycles():
    result = await PlanningAgent(InvalidLLM()).run("x", AgentContext("req"))
    assert not result.success
    assert result.data["status"] == "invalid_plan"
    assert "cycle" in result.error.lower()


class BrokenLLM:
    async def chat(self, request):
        raise RuntimeError("offline")


@pytest.mark.asyncio
async def test_planning_agent_handles_llm_failure():
    result = await PlanningAgent(BrokenLLM()).run("x", AgentContext("req"))
    assert not result.success
    assert result.data["status"] == "llm_failed"
