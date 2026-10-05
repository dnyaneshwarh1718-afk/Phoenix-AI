import pytest

from app.agents.base_agent import AgentContext
from app.agents.planning_agent import PlanningAgent


@pytest.mark.asyncio
async def test_planning_fast_path_avoids_llm_for_simple_analysis_goal():
    class ExplodingLLM:
        async def chat(self, request):
            raise AssertionError("LLM should not be called on deterministic planning fast path")

    agent = PlanningAgent(ExplodingLLM(), fast_path_enabled=True)
    result = await agent.run(
        "Create an execution plan for analyzing a sales dataset and identifying the top products by revenue.",
        AgentContext("req"),
    )

    assert result.success
    assert result.data["planner_mode"] == "deterministic_fast_path"
    assert result.data["step_count"] == 4
    assert result.data["performance_timings"]["planning_llm_calls"] == 0
