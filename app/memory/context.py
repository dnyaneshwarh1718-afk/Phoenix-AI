from __future__ import annotations

from app.memory.store import MemoryRecord, MemoryStore


def build_memory_context(
    store: MemoryStore,
    *,
    user_id: str,
    project_id: str | None,
    query: str,
    limit: int = 6,
) -> list[dict]:
    records = store.search(
        user_id=user_id,
        project_id=project_id,
        query=query,
        limit=limit,
    )
    return [
        {
            "memory_id": record.memory_id,
            "kind": record.kind,
            "content": record.content,
            "importance": record.importance,
            "created_at": record.created_at,
        }
        for record in records
    ]


def render_memory_context(records: list[dict]) -> str:
    if not records:
        return "No relevant prior memory was found."
    return "\n".join(
        f"- {item['content']}" for item in records
    )
