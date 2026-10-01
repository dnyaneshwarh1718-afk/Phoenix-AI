import uuid

from app.agents.application_agent import ApplicationControlAgent
from app.agents.base_agent import AgentContext
from app.agents.memory_agent import MemoryAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.rag_agent import RAGAgent
from app.agents.research_agent import ResearchAgent
from app.agents.vision_agent import ComputerVisionAgent
from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.llm.gateway import LLMGateway
from app.llm.models import ModelRequest
from app.orchestrator.router import IntentRouter
from app.rag.factory import build_rag_engine
from app.rag.rag_engine import RAGEngine
from app.research.search import DuckDuckGoSearchProvider
from app.workflows.graph import build_phoenix_graph

logger = get_logger(__name__)


class PhoenixOrchestrator:
    def __init__(self, llm: LLMGateway, rag_engine: RAGEngine | None = None, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()
        self.router = IntentRouter()
        self.rag_engine = rag_engine or build_rag_engine(self.settings)

        self.agents = {
            "planning": PlanningAgent(llm),
            "rag": RAGAgent(self.rag_engine),
            "research": ResearchAgent(
                llm,
                DuckDuckGoSearchProvider(endpoint=self.settings.research_search_endpoint),
                max_sources=self.settings.research_max_sources,
                timeout_seconds=self.settings.research_timeout_seconds,
            ),
            "application": ApplicationControlAgent(),
            "memory": MemoryAgent(),
            "vision": ComputerVisionAgent(),
        }
        self.graph = build_phoenix_graph(self)

    async def run(self, message: str, user_id: str = "default", project_id: str | None = None, metadata: dict | None = None) -> dict:
        if not message or not message.strip():
            raise ValueError("Message cannot be empty.")
        request_id = str(uuid.uuid4())
        state = {
            "request_id": request_id,
            "user_id": user_id,
            "project_id": project_id,
            "user_message": message.strip(),
            "metadata": dict(metadata or {}),
        }
        logger.info("Phoenix request started: request_id=%s", request_id)
        return await self.graph.ainvoke(state)

    async def execute_general(self, state: dict) -> dict:
        response = await self.llm.chat(
            ModelRequest(
                messages=[
                    {"role": "system", "content": "You are Phoenix AI, a practical engineering-focused AI assistant. Answer clearly and never claim an unavailable tool was executed."},
                    {"role": "user", "content": state["user_message"]},
                ],
                task_type="general",
            )
        )
        state["response"] = response.content
        state["metadata"].update({"provider": response.provider, "model": response.model})
        return state

    async def execute_specialized(self, state: dict) -> dict:
        agent_name = state["selected_agent"]
        if agent_name not in self.agents:
            return await self.execute_general(state)

        context = AgentContext(
            request_id=state["request_id"],
            user_id=state["user_id"],
            project_id=state.get("project_id"),
            metadata=state.get("metadata", {}),
        )
        result = await self.agents[agent_name].run(state["user_message"], context)
        state["response"] = result.content
        state["metadata"].update(result.data)
        if not result.success:
            state["error"] = result.error
        return state
