from app.orchestrator.router import IntentRouter


def test_document_reference_forces_rag():
    router = IntentRouter()
    assert router.classify("What is SQL?", {"document_reference": "SQL INTERVIEW QUESTIONS.docx"}) == "rag"


def test_document_language_routes_to_rag():
    router = IntentRouter()
    assert router.classify("According to the document, what is SQL?") == "rag"


def test_research_routes_to_research():
    router = IntentRouter()
    assert router.classify("Research the latest RAG techniques") == "research"
