from __future__ import annotations

import re
from typing import Iterable

from app.agents.base_agent import AgentContext, AgentResult, BaseAgent
from app.llm.gateway import LLMGateway
from app.llm.models import ModelRequest
from app.research.models import ResearchRequest, ResearchSource
from app.research.search import DuckDuckGoSearchProvider, SearchProvider


class ResearchAgent(BaseAgent):
    """External research agent with source-first, evidence-grounded synthesis."""

    name = "research"

    def __init__(
        self,
        llm: LLMGateway,
        search_provider: SearchProvider | None = None,
        *,
        max_sources: int = 6,
        timeout_seconds: float = 15.0,
    ):
        self.llm = llm
        self.search_provider = search_provider or DuckDuckGoSearchProvider()
        self.max_sources = max(1, max_sources)
        self.timeout_seconds = max(2.0, timeout_seconds)

    async def run(self, message: str, context: AgentContext) -> AgentResult:
        query = (message or "").strip()
        if not query:
            return AgentResult(False, error="Research query cannot be empty.")

        metadata = context.metadata or {}
        max_sources = self._bounded_int(metadata.get("research_max_sources"), self.max_sources, 1, 12)
        timeout = self._bounded_float(metadata.get("research_timeout_seconds"), self.timeout_seconds, 2.0, 60.0)

        try:
            sources = await self.search_provider.search(
                ResearchRequest(query=query, max_sources=max_sources, timeout_seconds=timeout)
            )
        except Exception as exc:
            return AgentResult(
                success=False,
                content="I could not complete the external research because the search provider was unavailable.",
                data={"status": "search_failed", "provider": getattr(self.search_provider, "name", "unknown"), "source_count": 0},
                error=f"Research search failed: {exc}",
            )

        if not sources:
            return AgentResult(
                success=True,
                content="I could not find usable external sources for this research query.",
                data={"status": "no_sources", "provider": getattr(self.search_provider, "name", "unknown"), "source_count": 0, "sources": []},
            )

        evidence = self._build_evidence(sources)
        prompt = self._build_prompt(query, evidence)

        try:
            response = await self.llm.chat(
                ModelRequest(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Phoenix AI's Research Agent. Answer only from the supplied research evidence. "
                                "Do not invent facts, URLs, citations, publication dates, or source claims. "
                                "If the evidence is insufficient or conflicting, say so explicitly. "
                                "Cite claims inline using [S1], [S2], etc. Keep the answer concise but useful."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    task_type="research",
                )
            )
        except Exception as exc:
            return AgentResult(
                success=False,
                content="Research sources were found, but synthesis failed.",
                data={"status": "synthesis_failed", "source_count": len(sources), "sources": self._serialize_sources(sources)},
                error=f"Research synthesis failed: {exc}",
            )

        content = (response.content or "").strip()
        citation_count = len(re.findall(r"\[S\d+\]", content))
        if citation_count == 0:
            # Never present an LLM synthesis as verified research when it ignored
            # the evidence/citation contract. The sources remain available to the
            # caller so a later retry or UI can recover without another search.
            return AgentResult(
                success=False,
                content="Research synthesis was not sufficiently source-grounded.",
                data={
                    "status": "unverified",
                    "provider": getattr(self.search_provider, "name", "unknown"),
                    "source_count": len(sources),
                    "sources": self._serialize_sources(sources),
                    "llm_provider": response.provider,
                    "llm_model": response.model,
                },
                error="Research model returned no source citations.",
            )

        return AgentResult(
            success=True,
            content=content,
            data={
                "status": "ok",
                "provider": getattr(self.search_provider, "name", "unknown"),
                "source_count": len(sources),
                "citation_count": citation_count,
                "sources": self._serialize_sources(sources),
                "llm_provider": response.provider,
                "llm_model": response.model,
            },
        )

    @staticmethod
    def _bounded_int(value, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(maximum, int(value))) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bounded_float(value, default: float, minimum: float, maximum: float) -> float:
        try:
            return max(minimum, min(maximum, float(value))) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_evidence(sources: Iterable[ResearchSource]) -> str:
        blocks = []
        for idx, source in enumerate(sources, 1):
            snippet = source.snippet.strip() or "No snippet supplied."
            blocks.append(f"[S{idx}] {source.title}\nURL: {source.url}\nDomain: {source.source_domain}\nEvidence: {snippet}")
        return "\n\n".join(blocks)

    @staticmethod
    def _build_prompt(query: str, evidence: str) -> str:
        return f"""Research question:\n{query}\n\nAvailable external evidence:\n{evidence}\n\nInstructions:\n1. Synthesize the answer from the evidence above.\n2. Put [S#] citations immediately after supported claims.\n3. Do not use knowledge that is not supported by the supplied evidence.\n4. Mention important uncertainty or source disagreement.\n5. End with a short 'Sources' list containing the cited source numbers and URLs."""

    @staticmethod
    def _serialize_sources(sources: Iterable[ResearchSource]) -> list[dict]:
        return [
            {"id": f"S{i}", "title": s.title, "url": s.url, "snippet": s.snippet, "domain": s.source_domain, "rank": s.rank}
            for i, s in enumerate(sources, 1)
        ]
