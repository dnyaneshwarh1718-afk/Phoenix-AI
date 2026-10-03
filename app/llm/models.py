from dataclasses import dataclass
from typing import Literal


Provider = Literal["ollama", "gemini"]


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, str]]

    provider: Provider | None = None
    model: str | None = None

    temperature: float | None = None
    max_tokens: int | None = None
    # Request provider-side structured JSON output when supported.
    json_mode: bool = False

    task_type: str = "general"


@dataclass(frozen=True)
class ModelResponse:
    content: str

    provider: str
    model: str

    usage: dict

    raw: object | None = None