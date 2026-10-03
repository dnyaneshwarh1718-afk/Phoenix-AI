import pytest

from app.agents.base_agent import AgentContext
from app.agents.planning_agent import PlanningAgent
from app.llm.models import ModelResponse


VALID_PLAN = '''{
  "goal": "Analyze a sales dataset",
  "steps": [
    {"id":"step_1","title":"Inspect the dataset","description":"Load and inspect columns, types, missing values, and basic quality.","agent":"general","depends_on":[],"tools":[],"approval_required":false}
  ],
  "blockers": [], "assumptions": [], "notes": []
}'''


class FakeLLM:
    def __init__(self, content=None, *, error=None):
        self.content = content
        self.error = error
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return ModelResponse(self.content, "ollama", "fake", {})


class SequenceLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        content = self.outputs.pop(0)
        return ModelResponse(content, "ollama", "fake", {})


@pytest.mark.asyncio
async def test_planning_agent_returns_valid_structured_plan():
    llm = FakeLLM(VALID_PLAN)
    result = await PlanningAgent(llm).run("Analyze a sales dataset", AgentContext("req"))
    assert result.success
    assert result.data["step_count"] == 1
    assert result.data["plan"]["steps"][0]["id"] == "step_1"
    assert llm.requests[0].temperature == 0.1
    assert llm.requests[0].json_mode is True


@pytest.mark.asyncio
async def test_planning_agent_rejects_cycles():
    cyclic = '''{"goal":"x","steps":[
      {"id":"step_1","title":"A","description":"A","agent":"planning","depends_on":["step_2"],"tools":[],"approval_required":false},
      {"id":"step_2","title":"B","description":"B","agent":"planning","depends_on":["step_1"],"tools":[],"approval_required":false}],
      "blockers":[],"assumptions":[],"notes":[]}'''
    result = await PlanningAgent(FakeLLM(cyclic)).run("x", AgentContext("req"))
    assert not result.success
    assert "cycle" in (result.error or "").lower()
    assert result.data["status"] == "invalid_plan"


@pytest.mark.asyncio
async def test_planning_agent_handles_llm_failure():
    result = await PlanningAgent(FakeLLM(error=RuntimeError("offline"))).run("x", AgentContext("req"))
    assert not result.success
    assert result.data["status"] == "llm_failed"


@pytest.mark.asyncio
async def test_planning_agent_repairs_transient_invalid_output():
    malformed = "not json"
    llm = SequenceLLM([malformed, VALID_PLAN])
    result = await PlanningAgent(llm).run("Analyze a sales dataset", AgentContext("req"))
    assert result.success
    assert result.data["step_count"] == 1
    assert len(llm.requests) == 2
    assert llm.requests[1].json_mode is True
