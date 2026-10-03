from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    project_id: str | None
    kind: str
    content: str
    source: str
    importance: float
    created_at: str
    metadata: dict[str, Any]


class MemoryStore:
    """Small, durable, local-first memory store.

    SQLite is deliberately used as the first production implementation:
    it requires no additional service, survives process restarts, supports
    transactions, and can later be replaced behind this interface by
    PostgreSQL/pgvector without changing the MemoryAgent contract.
    """

    def __init__(self, database_path: str = "data/phoenix_memory.db") -> None:
        self._lock = threading.RLock()
        self._is_memory = database_path == ":memory:"
        if self._is_memory:
            # Each sqlite3.connect(":memory:") call creates a different
            # database. Use a shared in-memory URI and keep one anchor
            # connection alive for the lifetime of this store.
            self.database_path = f"file:phoenix_memory_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._anchor_connection = self._open_connection(self.database_path, uri=True)
        else:
            self.database_path = str(Path(database_path))
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
            self._anchor_connection = None
        self._initialize()

    def _open_connection(self, database: str, *, uri: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(database, timeout=10.0, uri=uri)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _connect(self) -> sqlite3.Connection:
        return self._open_connection(self.database_path, uri=self._is_memory)

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_memories_scope
                    ON memories(user_id, project_id, active, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_hash
                    ON memories(user_id, project_id, content_hash, active);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    kind,
                    tokenize='unicode61'
                );
                """
            )

            # Repair FTS if an older database was created without rows in the
            # virtual table. INSERT OR IGNORE keeps initialization idempotent.
            rows = conn.execute(
                "SELECT memory_id, content, kind FROM memories WHERE active = 1"
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO memories_fts(memory_id, content, kind) VALUES (?, ?, ?)",
                    (row["memory_id"], row["content"], row["kind"]),
                )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can",
        "do", "does", "for", "from", "how", "i", "in", "is", "it",
        "me", "my", "of", "on", "or", "that", "the", "this", "to",
        "use", "what", "where", "which", "who", "with", "you",
    }

    @classmethod
    def _query_tokens(cls, query: str) -> list[str]:
        tokens = re.findall(r"[\w][\w.-]*", query.lower(), flags=re.UNICODE)
        meaningful = [
            token for token in tokens
            if len(token) >= 2 and token not in cls._STOPWORDS
        ]
        # Keep order while removing duplicates.
        return list(dict.fromkeys(meaningful[:12]))

    @classmethod
    def _fts_query(cls, query: str) -> str:
        tokens = cls._query_tokens(query)
        # Natural-language questions contain terms that are not present in
        # the stored fact (e.g. "vector database"). OR semantics allow
        # relevant memory to match on the meaningful terms that do overlap.
        return " OR ".join(f'"{token}"*' for token in tokens)

    def add(
        self,
        *,
        user_id: str,
        content: str,
        kind: str = "fact",
        project_id: str | None = None,
        source: str = "user",
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        user_id = user_id or "default"
        importance = max(0.0, min(1.0, float(importance)))
        content_hash = self._hash(content)
        now = self._now()
        memory_id = hashlib.sha256(
            f"{user_id}|{project_id or ''}|{kind}|{content_hash}".encode("utf-8")
        ).hexdigest()[:24]

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """SELECT * FROM memories
                   WHERE user_id = ? AND project_id IS ? AND content_hash = ?
                     AND kind = ? AND active = 1
                   LIMIT 1""",
                (user_id, project_id, content_hash, kind),
            ).fetchone()
            if existing:
                return self._row_to_record(existing)

            conn.execute(
                """INSERT INTO memories
                   (memory_id, user_id, project_id, kind, content, content_hash,
                    source, importance, created_at, updated_at, active, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (
                    memory_id,
                    user_id,
                    project_id,
                    kind,
                    content,
                    content_hash,
                    source,
                    importance,
                    now,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO memories_fts(memory_id, content, kind) VALUES (?, ?, ?)",
                (memory_id, content, kind),
            )
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            return self._row_to_record(row)

    def save_turn(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_response: str,
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        if user_message.strip():
            records.append(
                self.add(
                    user_id=user_id,
                    project_id=project_id,
                    kind="conversation_user",
                    content=user_message,
                    source="conversation",
                    importance=0.35,
                    metadata=metadata,
                )
            )
        if assistant_response.strip():
            # Avoid turning very large tool/RAG payloads into permanent memory.
            response = assistant_response.strip()[:6000]
            records.append(
                self.add(
                    user_id=user_id,
                    project_id=project_id,
                    kind="conversation_assistant",
                    content=response,
                    source="conversation",
                    importance=0.25,
                    metadata=metadata,
                )
            )
        return records

    def search(
        self,
        *,
        user_id: str,
        query: str,
        project_id: str | None = None,
        limit: int = 8,
    ) -> list[MemoryRecord]:
        query = query.strip()
        if not query:
            return []
        limit = max(1, min(50, int(limit)))
        fts_query = self._fts_query(query)

        with self._lock, self._connect() as conn:
            rows: list[sqlite3.Row]
            if fts_query:
                rows = conn.execute(
                    """SELECT m.*
                       FROM memories_fts f
                       JOIN memories m ON m.memory_id = f.memory_id
                       WHERE memories_fts MATCH ? AND m.user_id = ? AND m.active = 1
                         AND (m.project_id IS ? OR m.project_id IS NULL)
                       ORDER BY bm25(memories_fts), m.importance DESC, m.created_at DESC
                       LIMIT ?""",
                    (fts_query, user_id or "default", project_id, limit),
                ).fetchall()
            else:
                rows = []

            # FTS is complemented by token-level substring matching. This is
            # important for natural-language questions whose exact wording is
            # different from the stored memory.
            if len(rows) < limit:
                tokens = self._query_tokens(query)
                if tokens:
                    conditions = " OR ".join("lower(content) LIKE ?" for _ in tokens)
                    params = [user_id or "default", project_id, *[f"%{token}%" for token in tokens], limit]
                    extra = conn.execute(
                        f"""SELECT * FROM memories
                           WHERE user_id = ? AND active = 1
                             AND (project_id IS ? OR project_id IS NULL)
                             AND ({conditions})
                           ORDER BY importance DESC, created_at DESC
                           LIMIT ?""",
                        params,
                    ).fetchall()
                    seen = {row["memory_id"] for row in rows}
                    rows.extend(row for row in extra if row["memory_id"] not in seen)

            return [self._row_to_record(row) for row in rows[:limit]]

    def recent(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
        limit: int = 8,
    ) -> list[MemoryRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE user_id = ? AND active = 1
                     AND (project_id IS ? OR project_id IS NULL)
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id or "default", project_id, max(1, min(50, int(limit)))),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def forget(
        self,
        *,
        user_id: str,
        query: str,
        project_id: str | None = None,
        limit: int = 20,
    ) -> int:
        matches = self.search(user_id=user_id, query=query, project_id=project_id, limit=limit)
        if not matches:
            return 0
        ids = [record.memory_id for record in matches]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "UPDATE memories SET active = 0, updated_at = ? WHERE memory_id = ?",
                [(self._now(), memory_id) for memory_id in ids],
            )
            conn.executemany(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                [(memory_id,) for memory_id in ids],
            )
        return len(ids)

    def count(self, *, user_id: str = "default", project_id: str | None = None) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS count FROM memories
                   WHERE user_id = ? AND active = 1
                     AND (project_id IS ? OR project_id IS NULL)""",
                (user_id, project_id),
            ).fetchone()
            return int(row["count"])

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            kind=row["kind"],
            content=row["content"],
            source=row["source"],
            importance=float(row["importance"]),
            created_at=row["created_at"],
            metadata=metadata,
        )
