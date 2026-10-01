from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    snippet: str = ""
    source_domain: str = ""
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchRequest:
    query: str
    max_sources: int = 6
    timeout_seconds: float = 15.0


@dataclass
class ResearchResponse:
    query: str
    answer: str
    sources: list[ResearchSource] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    provider: str = ""

    @property
    def source_count(self) -> int:
        return len(self.sources)
