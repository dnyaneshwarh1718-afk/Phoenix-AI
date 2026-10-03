import asyncio
from pathlib import Path

from app.agents.application_agent import ApplicationAction, ApplicationControlAgent
from app.agents.base_agent import AgentContext
from app.core.exceptions import ApprovalRequired


class FakeExecutor:
    def __init__(self):
        self.actions = []

    def execute(self, action: ApplicationAction):
        self.actions.append(action)
        return {
            "status": "executed",
            "operation": action.operation,
            "application": action.application,
            "target": action.target,
        }


def test_application_agent_opens_excel():
    executor = FakeExecutor()
    agent = ApplicationControlAgent(executor=executor)

    result = asyncio.run(
        agent.run(
            "Open this Excel file",
            AgentContext("r1", metadata={"target": r"R:\data\sales.xlsx"}),
        )
    )

    assert result.success
    assert result.data["application_action"]["application"] == "excel"
    assert result.data["application_action"]["operation"] == "open"
    assert executor.actions[0].target == r"R:\data\sales.xlsx"


def test_application_agent_analyzes_excel():
    executor = FakeExecutor()
    agent = ApplicationControlAgent(executor=executor)

    result = asyncio.run(
        agent.run(
            "Analyze this Excel workbook",
            AgentContext("r1", metadata={"target": "sales.xlsx"}),
        )
    )

    assert result.success
    assert result.data["application_action"]["operation"] == "inspect"


def test_application_agent_rejects_unknown_application():
    agent = ApplicationControlAgent(executor=FakeExecutor())
    result = asyncio.run(agent.run("Open Photoshop", AgentContext("r1")))
    assert not result.success


def test_application_agent_does_not_execute_unsupported_close():
    class Executor:
        def execute(self, action):
            return {"status": "unsupported", "message": "Closing is disabled."}

    result = asyncio.run(
        ApplicationControlAgent(executor=Executor()).run(
            "Close Excel",
            AgentContext("r1"),
        )
    )
    assert result.success
    assert "disabled" in result.content.lower()


def test_application_agent_parses_bare_filename():
    executor = FakeExecutor()
    agent = ApplicationControlAgent(executor=executor)
    result = asyncio.run(agent.run("Open sales.xlsx", AgentContext("r1")))
    assert result.success
    assert executor.actions[0].target == "sales.xlsx"
