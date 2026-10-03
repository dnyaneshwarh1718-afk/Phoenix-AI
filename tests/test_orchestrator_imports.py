"""Regression tests for the orchestrator's application-control dependencies."""


def test_orchestrator_uses_approval_manager_from_execution_layer():
    from app.execution.approval import ApprovalManager as ExecutionApprovalManager
    from app.orchestrator import orchestrator

    assert orchestrator.ApprovalManager is ExecutionApprovalManager
