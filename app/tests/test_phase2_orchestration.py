import pytest

from app.agents.base_agent import AgentResult
from app.memory.store import MemoryStore
from app.workflows.plan_executor import PlanExecutor
from app.planning.models import Plan, PlanStep
from app.workflows.graph import _build_memory_update_node, _build_verify_node


class Settings:
    memory_auto_capture = True
    memory_max_results = 8


class FakeAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def run(self, message, context):
        self.calls.append(message)
        return self.result


class FakeOrchestrator:
    def __init__(self):
        self.settings = Settings()
        self.memory_store = MemoryStore(":memory:")
        self.agents = {
            "rag": FakeAgent(AgentResult(True, "RAG answer", {"status": "ok"})),
            "research": FakeAgent(AgentResult(True, "Research answer", {"status": "ok"})),
            "memory": FakeAgent(AgentResult(True, "stored", {"status": "stored"})),
            "application": FakeAgent(AgentResult(True, "blocked", {"status": "blocked", "execution": {"executed": False}})),
            "vision": FakeAgent(AgentResult(True, "vision", {"status": "ok"})),
        }

    async def execute_general(self, state):
        state["response"] = "general result"
        return state


@pytest.mark.asyncio
async def test_plan_executor_runs_dependencies_in_order():
    orch = FakeOrchestrator()
    plan = Plan(
        "test",
        (
            PlanStep("step_1", "one", "first", "rag"),
            PlanStep("step_2", "two", "second", "research", ("step_1",)),
        ),
    )
    state = {"request_id": "r", "user_id": "u", "metadata": {}, "memory_context": []}
    result = await PlanExecutor(orch).execute(plan, state)
    assert result["success"] is True
    assert [s["step_id"] for s in result["steps"]] == ["step_1", "step_2"]


@pytest.mark.asyncio
async def test_plan_executor_fails_closed_on_unverified_step():
    orch = FakeOrchestrator()
    orch.agents["rag"] = FakeAgent(AgentResult(True, "", {"status": "unverified"}))
    plan = Plan("test", (PlanStep("step_1", "one", "first", "rag"),))
    result = await PlanExecutor(orch).execute(plan, {"request_id": "r", "user_id": "u", "metadata": {}, "memory_context": []})
    assert result["success"] is False
    assert result["status"] == "unverified"


def test_phase2_verifier_accepts_safe_application_block():
    orch = FakeOrchestrator()
    node = _build_verify_node(orch)
    result = node({
        "intent": "application",
        "response": "blocked",
        "metadata": {"execution": {"status": "blocked", "executed": False}},
    })
    assert result["verification"]["verified"] is True


def test_phase2_memory_update_stores_successful_task_outcome():
    orch = FakeOrchestrator()
    node = _build_memory_update_node(orch)
    result = node({
        "intent": "research",
        "execute_plan": True,
        "user_id": "u",
        "user_message": "research task",
        "response": "completed",
        "request_id": "r",
        "metadata": {},
        "verification": {"verified": True},
    })
    assert result["metadata"]["memory_update"]["status"] == "stored"
    assert orch.memory_store.count(user_id="u") == 1
