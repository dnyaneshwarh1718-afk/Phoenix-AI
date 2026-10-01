from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextItem:
    rank: int
    chunk_id: str
    document_id: str
    text: str
    score: float
    retrieval_method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalContext:
    query: str
    items: list[ContextItem]
    context_text: str


class ContextBuilder:
    """Convert ranked retrieval results into bounded, citation-friendly evidence."""

    def __init__(self, max_chunks: int = 5, max_context_chars: int = 8000, max_chunk_chars: int = 3000) -> None:
        if max_chunks <= 0 or max_context_chars <= 0 or max_chunk_chars <= 0:
            raise ValueError("Context limits must be positive.")
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars

    def build(self, query: str, results) -> RetrievalContext:
        if not query or not query.strip() or not results:
            return RetrievalContext(query=query.strip() if query else "", items=[], context_text="")

        selected: list[ContextItem] = []
        seen: set[str] = set()
        current_chars = 0

        for rank, result in enumerate(results, start=1):
            chunk_id = str(getattr(result, "chunk_id", ""))
            if not chunk_id or chunk_id in seen or len(selected) >= self.max_chunks:
                continue
            text = str(getattr(result, "text", "") or "").strip()
            if not text:
                continue
            text = text[: self.max_chunk_chars].rstrip() + ("..." if len(text) > self.max_chunk_chars else "")
            item = ContextItem(
                rank=len(selected) + 1,
                chunk_id=chunk_id,
                document_id=str(getattr(result, "document_id", "")),
                text=text,
                score=float(getattr(result, "score", 0.0)),
                retrieval_method=str(getattr(result, "retrieval_method", "hybrid")),
                metadata=dict(getattr(result, "metadata", {}) or {}),
            )
            block = self._format_block(item)
            block_size = len(block) + 2
            if current_chars + block_size > self.max_context_chars:
                break
            selected.append(item)
            seen.add(chunk_id)
            current_chars += block_size

        context_text = "\n\n".join(self._format_block(item) for item in selected)
        return RetrievalContext(query=query.strip(), items=selected, context_text=context_text)

    @staticmethod
    def _format_block(item: ContextItem) -> str:
        location = []
        for key in ("page", "slide", "sheet"):
            if key in item.metadata:
                location.append(f"{key.title()}: {item.metadata[key]}")
        location_text = " | ".join(location)
        header = f"[Evidence {item.rank}]\nDocument ID: {item.document_id}\nChunk ID: {item.chunk_id}"
        if location_text:
            header += f"\n{location_text}"
        return f"{header}\n\n{item.text}"
