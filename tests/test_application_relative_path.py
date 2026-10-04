from app.agents.application_agent import ApplicationActionParser


def test_application_parser_accepts_relative_path():
    action = ApplicationActionParser().parse(
        "Inspect the Excel file evaluation/corpus/phoenix_evaluation.xlsx."
    )
    assert action.operation == "inspect"
    assert action.application == "excel"
    assert action.target == "evaluation/corpus/phoenix_evaluation.xlsx"
