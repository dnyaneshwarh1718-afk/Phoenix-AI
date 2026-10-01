from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityPolicy:
    require_approval_for_tools: bool = True
    max_tool_calls_per_task: int = 20


def is_high_risk_action(action: str) -> bool:
    risky = {
        "delete",
        "upload",
        "send_email",
        "execute_shell",
        "modify_production",
    }
    return action.lower() in risky
