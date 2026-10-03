import asyncio

from app.agents.application_agent import ApplicationAction, ApplicationControlAgent, WindowsApplicationExecutor
from app.agents.base_agent import AgentContext
from app.core.exceptions import ApprovalRequired
from app.core.security import SecurityPolicy
from app.execution.approval import ApprovalManager


def test_modify_is_blocked_before_executor_side_effect():
    manager = ApprovalManager(SecurityPolicy(require_approval_for_tools=True))
    executor = WindowsApplicationExecutor(manager)
    action = ApplicationAction(operation="modify", application="excel", target="sales.xlsx")
    try:
        executor.execute(action)
    except ApprovalRequired:
        pass
    else:
        raise AssertionError("modify must require explicit approval")


def test_application_control_respects_runtime_disable_flag():
    agent = ApplicationControlAgent(executor=object())
    result = asyncio.run(
        agent.run(
            "Open Excel",
            AgentContext("r1", metadata={"application_enabled": False}),
        )
    )
    assert not result.success
    assert result.data["application_action"]["status"] == "disabled"
