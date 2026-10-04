from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request):
    settings = request.app.state.settings
    return {"status": "ok", "service": settings.app_name, "environment": settings.app_env}


@router.get("/health/rag")
async def rag_health(request: Request):
    rag = request.app.state.orchestrator.rag_engine
    qdrant_ok = False
    bm25_ok = False
    try:
        qdrant_ok = bool(rag.hybrid_retriever.dense_retriever.vector_store.health_check())
    except Exception:
        pass
    try:
        bm25_ok = bool(rag.hybrid_retriever.keyword_retriever.bm25_store.health_check())
    except Exception:
        pass
    return {
        "status": "ok" if qdrant_ok and bm25_ok else "degraded",
        "qdrant": qdrant_ok,
        "bm25": bm25_ok,
        "retrieval_limit": rag.retrieval_limit,
        "candidate_limit": rag.candidate_limit,
    }
