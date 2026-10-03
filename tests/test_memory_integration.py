from app.agents.base_agent import AgentContext
from app.memory.store import MemoryStore
from app.workflows.graph import _build_memory_node


def test_memory_context_node_retrieves_user_scoped_memory():
    class Settings:
        memory_auto_capture = True
        memory_max_results = 8

    class Orchestrator:
        settings = Settings()
        memory_store = MemoryStore("memory_integration.db")

    orchestrator = Orchestrator()
    orchestrator.memory_store.add(user_id="alice", content="Phoenix AI uses Qdrant", kind="fact")
    node = _build_memory_node(orchestrator)
    state = {
        "request_id": "r1",
        "user_id": "alice",
        "user_message": "What vector database does Phoenix AI use?",
        "metadata": {},
    }
    result = node(state)
    assert result["memory_context"]
    assert result["memory_context"][0]["content"] == "Phoenix AI uses Qdrant"
