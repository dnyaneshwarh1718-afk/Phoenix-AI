from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.rag.retrieval.rrf import RRFResult


@dataclass(frozen=True)
class RerankedResult:
    """Final relevance-ranked retrieval candidate."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    retrieval_method: str = "rerank"
    metadata: dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 0.0
    lexical_score: float = 0.0


class HybridReranker:
    """
    Deterministic local reranker.

    It deliberately has no model download or external API dependency.
    RRF provides recall; this stage improves precision using query-term
    coverage, phrase matches, and the original RRF score.
    """

    _TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
    _STOP_WORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "why", "with", "do", "does", "did", "can",
        "could", "should", "would", "about", "into", "than", "then",
    })

    def __init__(self, *, rrf_weight: float = 0.35, lexical_weight: float = 0.65) -> None:
        if rrf_weight < 0 or lexical_weight < 0:
            raise ValueError("Reranker weights must be non-negative.")
        total = rrf_weight + lexical_weight
        if total <= 0:
            raise ValueError("At least one reranker weight must be positive.")
        self.rrf_weight = rrf_weight / total
        self.lexical_weight = lexical_weight / total

    def rerank(self, query: str, results: Iterable[Any], limit: int = 5) -> list[RerankedResult]:
        if not query or not query.strip() or limit <= 0:
            return []

        candidates = list(results or [])
        if not candidates:
            return []

        query_tokens = self._meaningful_tokens(query)
        query_phrase = " ".join(query_tokens)
        scored: list[RerankedResult] = []
        seen: set[str] = set()

        for result in candidates:
            chunk_id = self._chunk_id(result)
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)

            text = self._text(result)
            lexical = self._lexical_score(query_tokens, query_phrase, text)
            rrf = max(0.0, float(getattr(result, "score", 0.0)))
            # RRF values are normally small; squash them to a stable 0..1 range.
            rrf_norm = rrf / (rrf + 0.02) if rrf else 0.0
            final = self.rrf_weight * rrf_norm + self.lexical_weight * lexical

            scored.append(
                RerankedResult(
                    chunk_id=chunk_id,
                    document_id=str(getattr(result, "document_id", "")),
                    chunk_index=int(getattr(result, "chunk_index", -1)),
                    text=text,
                    score=final,
                    metadata=dict(getattr(result, "metadata", {}) or {}),
                    rrf_score=rrf,
                    lexical_score=lexical,
                )
            )

        scored.sort(key=lambda item: (-item.score, -item.rrf_score, item.chunk_id))
        return scored[:limit]

    @classmethod
    def _meaningful_tokens(cls, text: str) -> list[str]:
        return [
            token
            for token in cls._TOKEN_RE.findall(text.lower())
            if token not in cls._STOP_WORDS and len(token) > 1
        ]

    @classmethod
    def _lexical_score(cls, query_tokens: list[str], phrase: str, text: str) -> float:
        if not query_tokens or not text:
            return 0.0
        text_lower = text.lower()
        text_tokens = set(cls._TOKEN_RE.findall(text_lower))
        coverage = sum(token in text_tokens for token in query_tokens) / len(query_tokens)
        phrase_bonus = 0.20 if phrase and phrase in text_lower else 0.0
        title_bonus = 0.0
        # Metadata title/file name can be disproportionately useful for short queries.
        return min(1.0, 0.80 * coverage + phrase_bonus + title_bonus)

    @staticmethod
    def _chunk_id(result: Any) -> str:
        chunk_id = getattr(result, "chunk_id", None)
        if chunk_id:
            return str(chunk_id)
        chunk = getattr(result, "chunk", None)
        return str(getattr(chunk, "chunk_id", "")) if chunk else ""

    @staticmethod
    def _text(result: Any) -> str:
        text = getattr(result, "text", None)
        if text is None:
            chunk = getattr(result, "chunk", None)
            text = getattr(chunk, "text", "") if chunk else ""
        return str(text or "").strip()
