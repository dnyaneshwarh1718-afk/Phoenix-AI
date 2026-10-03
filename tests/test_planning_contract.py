import pytest

from app.agents.base_agent import AgentContext
from app.agents.planning_agent import PlanningAgent
from app.llm.models import ModelResponse
from app.orchestrator.orchestrator import PhoenixOrchestrator


VALID_PLAN = """{
  "goal": "Analyze a sales dataset",
  "steps": [
    {
      "id": "step_1",
      "title": "Inspect the dataset",
      "description": "Load and inspect columns, types, missing values, and basic quality.",
      "agent": "general",
      "depends_on": [],
      "tools": [],
      "approval_required": false
    }
  ],
  "blockers": [],
  "assumptions": [],
  "notes": []
}"""


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        return ModelResponse(
            content=self.content,
            provider="ollama",
            model="fake",
            usage={},
        )


@pytest.mark.asyncio
async def test_planning_agent_exposes_step_count_and_preserves_temperature():
    llm = FakeLLM(VALID_PLAN)
    result = await PlanningAgent(llm).run(
        "Analyze a sales dataset",
        AgentContext("req"),
    )

    assert result.success
    assert result.data["step_count"] == 1
    assert len(result.data["plan"]["steps"]) == 1
    assert llm.requests[0].temperature == 0.1


@pytest.mark.asyncio
async def test_invalid_plan_fails_closed():
    cyclic_plan = """{
      "goal": "x",
      "steps": [
        {"id":"step_1","title":"A","description":"A","agent":"planning","depends_on":["step_2"],"tools":[],"approval_required":false},
        {"id":"step_2","title":"B","description":"B","agent":"planning","depends_on":["step_1"],"tools":[],"approval_required":false}
      ],
      "blockers": [], "assumptions": [], "notes": []
    }"""

    result = await PlanningAgent(FakeLLM(cyclic_plan)).run(
        "x",
        AgentContext("req"),
    )

    assert not result.success
    assert "cycle" in (result.error or "").lower()
    assert result.data["status"] == "invalid_plan"


def test_final_planning_contract_reconstructs_step_count():
    state = {
        "intent": "planning",
        "plan": "Goal: test",
        "metadata": {
            "planning": True,
            "plan": {"steps": [{"id": "step_1"}, {"id": "step_2"}]},
        },
    }

    PhoenixOrchestrator._finalize_planning_contract(state)

    assert state["metadata"]["step_count"] == 2


def test_final_planning_contract_does_not_invent_count_on_failure():
    state = {
        "intent": "planning",
        "plan": "The planning model returned an invalid execution plan.",
        "metadata": {"planning": True, "status": "invalid_plan"},
    }

    PhoenixOrchestrator._finalize_planning_contract(state)

    assert "step_count" not in state["metadata"]
