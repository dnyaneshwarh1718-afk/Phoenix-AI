from app.orchestrator.router import IntentRouter


def test_rag_intent():
    router = IntentRouter()
    assert router.classify("Find information in this PDF") == "rag"


def test_research_intent():
    router = IntentRouter()
    assert router.classify("Research the latest RAG techniques") == "research"


def test_application_intent():
    router = IntentRouter()
    assert router.classify("Open this Excel file and analyze it") == "application"


def test_general_intent():
    router = IntentRouter()
    assert router.classify("What is machine learning?") == "general"
