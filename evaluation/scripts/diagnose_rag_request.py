from __future__ import annotations
import argparse, json, os
from pathlib import Path
import httpx

API = os.getenv("PHOENIX_API_URL", "http://127.0.0.1:8000")
QDRANT = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("QDRANT_COLLECTION", "phoenix_documents")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", default="evaluation/corpus/phoenix_architecture.txt")
    parser.add_argument("--question", default="What vector database does Phoenix AI use?")
    args = parser.parse_args()

    root = Path.cwd()
    path = Path(args.document).resolve()
    print("=== PHOENIX RAG DIAGNOSTIC ===")
    print("API:", API)
    print("Document:", path)
    print("Question:", args.question)

    with httpx.Client(timeout=120) as c:
        reg = c.post(f"{API}/api/v1/chat", json={
            "message": args.question,
            "document_reference": str(path),
            "auto_index": True,
            "user_id": "e2e-diagnostic",
        }, timeout=300)
        print("\nAPI STATUS:", reg.status_code)
        try:
            body = reg.json()
            print(json.dumps(body, indent=2, ensure_ascii=False)[:12000])
        except Exception:
            print(reg.text)

        emb = c.post(f"{OLLAMA}/api/embed", json={"model": MODEL, "input": args.question})
        emb.raise_for_status()
        vector = emb.json()["embeddings"][0]

        # Global retrieval proves the vector/index layer.
        global_q = c.post(
            f"{QDRANT}/collections/{COLLECTION}/points/query",
            json={"query": vector, "limit": 5, "with_payload": True},
        )
        global_q.raise_for_status()
        points = global_q.json().get("result", {}).get("points", [])
        print("\nGLOBAL QDRANT TOP RESULTS:")
        for p in points[:5]:
            payload = p.get("payload") or {}
            print(
                " score=", p.get("score"),
                " document_id=", payload.get("document_id"),
                " source=", (payload.get("metadata") or {}).get("source_path"),
            )

        source_filter = {
            "query": vector,
            "limit": 5,
            "with_payload": True,
            "query_filter": {
                "must": [{
                    "key": "metadata.source_path",
                    "match": {"value": str(path)}
                }]
            },
        }
        scoped = c.post(
            f"{QDRANT}/collections/{COLLECTION}/points/query",
            json=source_filter,
        )
        print("\nSOURCE-PATH SCOPED STATUS:", scoped.status_code)
        if scoped.is_success:
            pts = scoped.json().get("result", {}).get("points", [])
            print("SOURCE-PATH SCOPED COUNT:", len(pts))
            for p in pts[:5]:
                payload = p.get("payload") or {}
                print(
                    " score=", p.get("score"),
                    " document_id=", payload.get("document_id"),
                    " source=", (payload.get("metadata") or {}).get("source_path"),
                )

if __name__ == "__main__":
    main()
