from app.orchestrator.router import IntentRouter


def test_memory_store_request_overrides_project_knowledge_keywords():
    router = IntentRouter()
    assert router.classify(
        "Remember that my primary Phoenix AI vector database is Qdrant."
    ) == "memory"


def test_memory_recall_request_overrides_project_knowledge_keywords():
    router = IntentRouter()
    assert router.classify(
        "What vector database did I say is my primary choice for Phoenix AI?"
    ) == "memory"
