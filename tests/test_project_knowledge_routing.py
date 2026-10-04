from app.orchestrator.router import IntentRouter


def test_phoenix_vector_database_question_routes_to_rag():
    assert IntentRouter().classify("What vector database does Phoenix AI use?") == "rag"


def test_explicit_document_question_routes_to_rag():
    assert IntentRouter().classify("According to the Phoenix AI documentation, what vector database is used?") == "rag"


def test_application_action_keeps_application_precedence():
    assert IntentRouter().classify("Open the Phoenix AI Excel workbook") == "application"


def test_unrelated_general_question_stays_general():
    assert IntentRouter().classify("What is the capital of France?") == "general"
