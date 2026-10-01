import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.mark.parametrize("message", [
    "Plan how to build a production RAG pipeline",
    "Create an execution plan for analyzing a sales dataset",
])
def test_live_planning_api(message):
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": message})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"] == "planning"
    assert body["selected_agent"] == "planning"
    assert body["response"]
    assert body["plan"]
    assert body["metadata"]["step_count"] >= 1
    assert body["metadata"]["status"] in {"ready", "blocked"}
