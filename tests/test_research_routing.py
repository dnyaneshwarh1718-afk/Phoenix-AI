from app.orchestrator.router import IntentRouter


def test_research_phrases_route_to_research():
    router = IntentRouter()
    for q in [
        "Research the latest RAG techniques",
        "search web for the latest Python release",
        "investigate current Qdrant features",
        "look up the latest Ollama documentation",
    ]:
        assert router.classify(q) == "research"


def test_planning_phrases_route_to_planning():
    router = IntentRouter()
    for q in [
        "plan how to build a RAG pipeline",
        "create an execution plan for my data science project",
        "break this project into steps",
    ]:
        assert router.classify(q) == "planning"
