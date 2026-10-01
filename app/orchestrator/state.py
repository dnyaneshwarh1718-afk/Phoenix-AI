from typing import Any, TypedDict


class PhoenixState(TypedDict, total=False):
    request_id: str
    user_id: str
    project_id: str | None
    user_message: str
    intent: str
    selected_agent: str
    plan: str
    response: str
    error: str | None
    metadata: dict[str, Any]
