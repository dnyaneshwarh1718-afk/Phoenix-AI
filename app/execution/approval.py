from app.core.exceptions import ApprovalRequired
from app.core.security import SecurityPolicy, is_high_risk_action


class ApprovalManager:
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy

    def check(self, action: str, approved: bool = False) -> None:
        if not self.policy.require_approval_for_tools:
            return

        if is_high_risk_action(action) and not approved:
            raise ApprovalRequired(
                f"Human approval required before high-risk action: {action}"
            )
