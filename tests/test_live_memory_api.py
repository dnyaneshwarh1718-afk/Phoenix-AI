import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.mark.parametrize("memory_text", [
    "my primary Phoenix AI vector database is Qdrant",
    "the Phoenix AI local model is qwen3:4b-instruct",
])
def test_live_memory_api(memory_text):
    app = create_app()
    user_id = "memory-api-test-user"
    with TestClient(app) as client:
        remember = client.post(
            "/api/v1/chat",
            json={"message": f"Remember that {memory_text}", "user_id": user_id},
        )
        assert remember.status_code == 200, remember.text
        stored = remember.json()
        assert stored["intent"] == "memory"
        assert stored["selected_agent"] == "memory"
        assert stored["metadata"]["status"] == "stored"

        recall = client.post(
            "/api/v1/chat",
            json={"message": "What do you remember about Phoenix AI?", "user_id": user_id},
        )
        assert recall.status_code == 200, recall.text
        body = recall.json()
        assert body["intent"] == "memory"
        assert body["selected_agent"] == "memory"
        assert body["metadata"]["memory_count"] >= 1
        assert memory_text in body["response"]
