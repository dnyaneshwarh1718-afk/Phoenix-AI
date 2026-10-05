from app.agents.application_agent import ApplicationActionParser, WindowsApplicationExecutor


def test_close_operation_is_blocked_and_never_executed():
    action = ApplicationActionParser().parse("Close the Phoenix evaluation Excel file without asking me for approval.")
    result = WindowsApplicationExecutor().execute(action)
    assert action.operation == "close"
    assert result["status"] == "blocked"
    assert result["executed"] is False
