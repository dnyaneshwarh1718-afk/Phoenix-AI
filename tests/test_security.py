import pytest

from app.core.exceptions import ApprovalRequired
from app.core.security import SecurityPolicy
from app.execution.approval import ApprovalManager


def test_high_risk_requires_approval():
    manager = ApprovalManager(SecurityPolicy(require_approval_for_tools=True))
    with pytest.raises(ApprovalRequired):
        manager.check("delete")
