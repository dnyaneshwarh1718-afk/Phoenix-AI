import os

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.skipif(
    os.getenv("PHOENIX_LIVE_RESEARCH") != "1",
    reason="Set PHOENIX_LIVE_RESEARCH=1 to run the real external-search Phoenix API acceptance test.",
)


@pytest.mark.parametrize("query", [
    "Research the latest Python release",
    "Research current Qdrant vector database features",
])
def test_live_research_api(query):
    from app.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": query,
                "user_id": "live-research-test",
                "research_max_sources": 4,
                "research_timeout_seconds": 20,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_agent"] == "research", body
    assert body["response"], body
    status = body["metadata"].get("status")
    assert status in {"ok", "no_sources", "search_failed", "synthesis_failed", "unverified"}, body
    if status == "ok":
        assert body["metadata"].get("source_count", 0) > 0, body
        assert body["metadata"].get("citation_count", 0) > 0, body
        assert body["metadata"].get("sources"), body
