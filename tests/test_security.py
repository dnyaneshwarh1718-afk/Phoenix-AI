import pytest

from app.core.exceptions import ApprovalRequired
from app.core.security import SecurityPolicy
from app.execution.approval import ApprovalManager


def test_high_risk_requires_approval():
    manager = ApprovalManager(SecurityPolicy(require_approval_for_tools=True))
    with pytest.raises(ApprovalRequired):
        manager.check("delete")


def test_modify_requires_approval():
    manager = ApprovalManager(SecurityPolicy(require_approval_for_tools=True))
    with pytest.raises(ApprovalRequired):
        manager.check("modify")


def test_safe_action_does_not_require_approval():
    manager = ApprovalManager(SecurityPolicy(require_approval_for_tools=True))
    manager.check("open")
