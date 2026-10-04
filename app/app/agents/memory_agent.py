from __future__ import annotations

import re

from app.agents.base_agent import AgentContext, AgentResult, BaseAgent
from app.memory.store import MemoryRecord, MemoryStore


class MemoryAgent(BaseAgent):
    """Persistent, user-scoped memory agent.

    Supported intents:
      - remember <fact>
      - forget <fact>
      - what do you remember about <topic>
      - natural-language memory lookup
    """

    name = "memory"

    def __init__(self, store: MemoryStore | None = None, *, max_results: int = 8):
        self.store = store or MemoryStore()
        self.max_results = max(1, min(20, max_results))

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        text = (message or "").strip()
        if not text:
            return AgentResult(False, error="Memory request cannot be empty.")

        user_id = context.user_id or "default"
        project_id = context.project_id
        metadata = context.metadata or {}

        try:
            if self._is_forget(text):
                query = self._strip_command(text, "forget")
                if not query:
                    return AgentResult(False, content="Tell me what you want me to forget.", data={"status": "invalid"})
                count = self.store.forget(user_id=user_id, project_id=project_id, query=query, limit=self.max_results)
                return AgentResult(
                    True,
                    content=(f"I forgot {count} matching memory item{'s' if count != 1 else ''}." if count else "I couldn't find a matching memory to forget."),
                    data={"status": "forgotten" if count else "not_found", "forgotten_count": count},
                )

            if self._is_remember(text):
                content = self._strip_command(text, "remember")
                if not content:
                    return AgentResult(False, content="Tell me what you want me to remember.", data={"status": "invalid"})
                record = self.store.add(
                    user_id=user_id,
                    project_id=project_id,
                    content=content,
                    kind="fact",
                    source="explicit_user_memory",
                    importance=0.95,
                    metadata={"request_id": context.request_id},
                )
                return AgentResult(
                    True,
                    content="I'll remember that.",
                    data={"status": "stored", "memory_id": record.memory_id, "memory_count": self.store.count(user_id=user_id, project_id=project_id)},
                )

            query = self._extract_query(text)
            records = self.store.search(
                user_id=user_id,
                project_id=project_id,
                query=query,
                limit=self.max_results,
            )
            if not records:
                records = self.store.recent(
                    user_id=user_id,
                    project_id=project_id,
                    limit=min(self.max_results, 5),
                ) if self._asks_recent(text) else []

            return AgentResult(
                True,
                content=self._render_records(records),
                data={
                    "status": "found" if records else "not_found",
                    "memory_count": len(records),
                    "memories": [self._serialize(record) for record in records],
                    "query": query,
                },
            )
        except Exception as exc:
            return AgentResult(
                False,
                content="I couldn't access persistent memory right now.",
                data={"status": "storage_error"},
                error=f"Memory operation failed: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _is_remember(text: str) -> bool:
        return bool(re.match(r"^remember(?: that)?\b", text, re.IGNORECASE))

    @staticmethod
    def _is_forget(text: str) -> bool:
        return bool(re.match(r"^forget(?: that)?\b", text, re.IGNORECASE))

    @staticmethod
    def _strip_command(text: str, command: str) -> str:
        return re.sub(rf"^\s*{re.escape(command)}(?:\s+that)?\s*[:,-]?\s*", "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def _extract_query(text: str) -> str:
        patterns = [
            r"what do you remember about\s+(.+)",
            r"what do you know about\s+(.+)",
            r"do you remember\s+(.+)",
            r"recall\s+(.+)",
            r"memory\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(" ?.")
        return text.strip(" ?.")

    @staticmethod
    def _asks_recent(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in ("recent", "latest", "last conversation", "last time"))

    @staticmethod
    def _serialize(record: MemoryRecord) -> dict:
        return {
            "memory_id": record.memory_id,
            "kind": record.kind,
            "content": record.content,
            "source": record.source,
            "importance": record.importance,
            "created_at": record.created_at,
            "project_id": record.project_id,
        }

    @classmethod
    def _render_records(cls, records: list[MemoryRecord]) -> str:
        if not records:
            return "I don't have any matching memories yet."
        lines = ["Here is what I remember:"]
        for index, record in enumerate(records, 1):
            lines.append(f"{index}. {record.content}")
        return "\n".join(lines)
