from fastapi import APIRouter, Request
from app.api.schemas import ChatRequest, ChatResponse


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    orchestrator = request.app.state.orchestrator
    result = await orchestrator.run(
        message=payload.message,
        user_id=payload.user_id,
        project_id=payload.project_id,
        metadata={
            "document_reference": payload.document_reference,
            "search_roots": payload.search_roots,
            "auto_index": payload.auto_index,
            "research_max_sources": payload.research_max_sources,
            "research_timeout_seconds": payload.research_timeout_seconds,
        },
    )

    return ChatResponse(
        request_id=result["request_id"],
        intent=result.get("intent", "unknown"),
        selected_agent=result.get("selected_agent", "unknown"),
        response=result.get("response", ""),
        plan=result.get("plan"),
        metadata=result.get("metadata", {}),
    )
