import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.skipif(
    os.getenv("PHOENIX_LIVE_RAG") != "1",
    reason="Set PHOENIX_LIVE_RAG=1 to run the real Ollama/Qdrant API acceptance test.",
)


@pytest.mark.parametrize(
    "query",
    ["What is SQL?", "Explain SQL joins."],
)
def test_live_rag_api(query):
    from app.api.main import create_app

    document = os.getenv("PHOENIX_RAG_DOCUMENT", r"R:\SQL INTERVIEW QUESTIONS.docx")
    assert Path(document).exists(), f"RAG document does not exist: {document}"

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": query,
                "user_id": "live-rag-test",
                "document_reference": document,
                "auto_index": True,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_agent"] == "rag", body
    assert body["response"], body
    assert body["metadata"].get("status") in {"ok", "unverified", "no_retrieval", "no_context", "not_found", "ambiguous", "indexed", "discovered_not_indexed"}, body
    assert body["metadata"].get("document_id"), body
