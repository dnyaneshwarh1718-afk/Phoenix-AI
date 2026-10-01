from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    request_id: str
    user_id: str = "default"
    project_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    success: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, message: str, context: AgentContext) -> AgentResult:
        raise NotImplementedError
