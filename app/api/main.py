from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logger import configure_logging
from app.llm.gateway import LLMGateway
from app.orchestrator.orchestrator import PhoenixOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    llm = LLMGateway(settings)
    app.state.settings = settings
    app.state.llm = llm
    app.state.orchestrator = PhoenixOrchestrator(llm, settings=settings)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Phoenix AI",
        version="0.1.0",
        description="Phoenix AI agentic AI platform foundation.",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api/v1")
    return app


app = create_app()
