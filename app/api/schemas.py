from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "default"
    project_id: str | None = None
    document_reference: str | None = None
    search_roots: list[str] | None = None
    auto_index: bool = True
    research_max_sources: int = Field(default=6, ge=1, le=12)
    research_timeout_seconds: float = Field(default=15.0, ge=2.0, le=60.0)


class ChatResponse(BaseModel):
    request_id: str
    intent: str
    selected_agent: str
    response: str
    plan: str | None = None
    execution_trace: list[dict] = Field(default_factory=list)
    verification: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
