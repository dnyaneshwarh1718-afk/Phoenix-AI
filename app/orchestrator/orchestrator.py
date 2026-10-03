import uuid

from app.agents.application_agent import ApplicationControlAgent
from app.agents.base_agent import AgentContext
from app.agents.memory_agent import MemoryAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.rag_agent import RAGAgent
from app.agents.research_agent import ResearchAgent
from app.agents.vision_agent import ComputerVisionAgent
from app.memory.store import MemoryStore
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
        self.memory_store = MemoryStore(self.settings.memory_database_path)
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
            "memory": MemoryAgent(self.memory_store, max_results=self.settings.memory_max_results),
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
        result = await self.graph.ainvoke(state)

        # The orchestrator owns the final API contract. Keep this boundary
        # normalization independent from individual graph nodes so a graph
        # refactor cannot silently drop planning metadata.
        if result.get("intent") == "planning":
            self._finalize_planning_contract(result)

        if self.settings.memory_auto_capture:
            self._capture_turn(result)

        return result

    @staticmethod
    def _finalize_planning_contract(state: dict) -> None:
        """Normalize the stable planning response contract in-place."""
        metadata = state.setdefault("metadata", {})
        plan = metadata.get("plan")

        step_count = metadata.get("step_count")
        if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 0:
            step_count = None

        if step_count is None and isinstance(plan, dict):
            steps = plan.get("steps")
            if isinstance(steps, (list, tuple)):
                step_count = len(steps)

        # Never invent a step count for a failed/invalid plan.
        if step_count is not None:
            metadata["step_count"] = step_count

        metadata.setdefault("planning", True)
        metadata.setdefault("status", "ready")

    def _capture_turn(self, state: dict) -> None:
        """Persist the completed interaction without affecting the response path."""
        try:
            self.memory_store.save_turn(
                user_id=state.get("user_id", "default"),
                project_id=state.get("project_id"),
                user_message=state.get("user_message", ""),
                assistant_response=state.get("response", ""),
                metadata={
                    "request_id": state.get("request_id"),
                    "intent": state.get("intent"),
                    "selected_agent": state.get("selected_agent"),
                },
            )
        except Exception as exc:
            # Memory must never make an otherwise successful agent request fail.
            logger.warning("Memory capture failed: %s", exc)

    async def execute_general(self, state: dict) -> dict:
        memory_context = state.get("memory_context") or []
        memory_text = "\n".join(
            f"- {item.get('content', '')}" for item in memory_context
        ) or "No relevant prior memory was found."
        response = await self.llm.chat(
            ModelRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Phoenix AI, a practical engineering-focused AI assistant. "
                            "Answer clearly and never claim an unavailable tool was executed. "
                            "Use the supplied memory only as prior user context; do not treat it "
                            "as authoritative external evidence. If memory is irrelevant, ignore it.\n\n"
                            f"Relevant prior memory:\n{memory_text}"
                        ),
                    },
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
            memory_context=state.get("memory_context", []),
        )

        result = await self.agents[agent_name].run(
            state["user_message"],
            context,
        )

        state["response"] = result.content

        # --------------------------------------------------
        # Merge agent result data
        # --------------------------------------------------

        result_data = dict(result.data or {})

        state["metadata"].update(result_data)

        # --------------------------------------------------
        # Agent failure handling
        # --------------------------------------------------

        if not result.success:
            state["error"] = result.error

        return state
