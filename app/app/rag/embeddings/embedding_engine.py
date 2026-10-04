
from __future__ import annotations

import httpx


class EmbeddingEngine:
    """Ollama embedding client. Generation and embeddings remain separate models."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": inputs},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                f"Ollama returned no embeddings: {data}"
            )
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def dimension(self) -> int:
        return len(self.embed_query("Phoenix AI embedding dimension test"))
