import pytest

from app.agents.base_agent import AgentResult
from app.memory.store import MemoryStore
from app.planning.models import Plan, PlanStep
from app.workflows.plan_executor import PlanExecutor
from app.workflows.graph import _build_verify_node, _build_memory_update_node


class Settings:
    memory_auto_capture = True
    memory_max_results = 8


class FakeAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def run(self, message, context):
        self.calls.append((message, context.metadata))
        return self.result


class FakeOrchestrator:
    def __init__(self):
        self.settings = Settings()
        self.memory_store = MemoryStore(":memory:")
        self.agents = {
            "rag": FakeAgent(AgentResult(True, "RAG answer", {"status": "ok"})),
            "research": FakeAgent(AgentResult(True, "Research answer", {"status": "ok"})),
            "memory": FakeAgent(AgentResult(True, "stored", {"status": "stored"})),
            "application": FakeAgent(AgentResult(True, "inspected", {"status": "ok", "execution": {"status": "executed", "executed": True}})),
            "vision": FakeAgent(AgentResult(True, "vision", {"status": "ok"})),
        }

    async def execute_general(self, state):
        state["response"] = "general result"
        return state


@pytest.mark.asyncio
async def test_downstream_step_receives_upstream_verified_result():
    orch = FakeOrchestrator()
    plan = Plan(
        "closed loop",
        (
            PlanStep("step_1", "inspect", "inspect", "rag"),
            PlanStep("step_2", "use result", "use the previous result", "research", ("step_1",)),
        ),
    )
    result = await PlanExecutor(orch).execute(
        plan,
        {"request_id": "r", "user_id": "u", "metadata": {}, "memory_context": []},
    )
    assert result["success"] is True
    prior = orch.agents["research"].calls[0][1]["plan_prior_results"]
    assert prior[0]["step_id"] == "step_1"
    assert prior[0]["success"] is True


@pytest.mark.asyncio
async def test_failed_dependency_blocks_downstream_steps():
    orch = FakeOrchestrator()
    orch.agents["rag"] = FakeAgent(AgentResult(False, "failed", {"status": "failed"}, "boom"))
    plan = Plan(
        "dependency failure",
        (
            PlanStep("step_1", "fail", "fail", "rag"),
            PlanStep("step_2", "dependent", "must not execute", "research", ("step_1",)),
        ),
    )
    result = await PlanExecutor(orch).execute(
        plan,
        {"request_id": "r", "user_id": "u", "metadata": {}, "memory_context": []},
    )
    assert result["success"] is False
    assert result["status"] == "failed"
    assert "step_2" in result["blocked_steps"]
    assert orch.agents["research"].calls == []


def test_verified_task_outcome_is_written_to_memory():
    orch = FakeOrchestrator()
    node = _build_memory_update_node(orch)
    result = node({
        "intent": "research",
        "execute_plan": True,
        "user_id": "u",
        "project_id": "p",
        "user_message": "research task",
        "response": "completed",
        "request_id": "r",
        "metadata": {},
        "verification": {"verified": True},
    })
    assert result["metadata"]["memory_update"]["status"] == "stored"
    assert orch.memory_store.count(user_id="u", project_id="p") == 1


def test_safety_block_is_verified_as_safe():
    orch = FakeOrchestrator()
    node = _build_verify_node(orch)
    result = node({
        "intent": "application",
        "response": "blocked",
        "metadata": {"execution": {"status": "blocked", "executed": False}},
    })
    assert result["verification"]["verified"] is True

@pytest.mark.asyncio
async def test_plan_executor_propagates_document_reference_to_application_step():
    orch = FakeOrchestrator()
    plan = Plan(
        "inspect provided workbook",
        (PlanStep("step_1", "inspect workbook", "Inspect the provided sales dataset.", "application"),),
    )
    result = await PlanExecutor(orch).execute(
        plan,
        {"request_id": "r", "user_id": "u", "metadata": {"document_reference": "evaluation/corpus/phoenix_evaluation.xlsx"}, "memory_context": []},
    )
    assert result["success"] is True
    assert orch.agents["application"].calls[0][1]["target"] == "evaluation/corpus/phoenix_evaluation.xlsx"


def test_document_scoped_analytical_workflow_routes_to_planning():
    from app.orchestrator.router import IntentRouter
    router = IntentRouter()
    intent = router.classify(
        "Analyze the sales dataset, identify the highest revenue product, and summarize the result.",
        {"document_reference": "evaluation/corpus/phoenix_evaluation.xlsx"},
    )
    assert intent == "planning"


def test_document_scoped_knowledge_question_remains_rag():
    from app.orchestrator.router import IntentRouter
    router = IntentRouter()
    intent = router.classify(
        "According to the Phoenix evaluation spreadsheet, which product has the highest revenue?",
        {"document_reference": "evaluation/corpus/phoenix_evaluation.xlsx"},
    )
    assert intent == "rag"
