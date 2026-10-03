from app.orchestrator.router import IntentRouter


def test_open_excel_file_routes_to_application():
    assert IntentRouter().classify("Open this Excel file and analyze it") == "application"


def test_excel_knowledge_question_routes_to_rag():
    assert IntentRouter().classify("According to this Excel document, what is the revenue?") == "rag"


def test_document_reference_overrides_application_keyword():
    assert IntentRouter().classify(
        "What is in Excel?",
        {"document_reference": "sales.xlsx"},
    ) == "rag"


def test_power_bi_operation_routes_to_application():
    assert IntentRouter().classify("Launch Power BI and open the report") == "application"


def test_jupyter_operation_routes_to_application():
    assert IntentRouter().classify("Run this Jupyter notebook") == "application"
