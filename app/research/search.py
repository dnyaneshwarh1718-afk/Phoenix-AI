from __future__ import annotations

import asyncio
import html
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx

from app.research.models import ResearchRequest, ResearchSource


class SearchProvider(ABC):
    name = "base"

    @abstractmethod
    async def search(self, request: ResearchRequest) -> list[ResearchSource]:
        raise NotImplementedError


class _DuckDuckGoParser(HTMLParser):
    """Small stdlib-only parser for DDG HTML results; no BeautifulSoup dependency."""

    def __init__(self):
        super().__init__()
        self.results: list[ResearchSource] = []
        self._current_url = ""
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._mode: str | None = None
        self._rank = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._current_url = self._normalize_url(attrs.get("href", ""))
            self._current_title = []
            self._current_snippet = []
            self._mode = "title"
        elif "result__snippet" in classes:
            self._mode = "snippet"

    def handle_data(self, data):
        if self._mode == "title":
            self._current_title.append(data)
        elif self._mode == "snippet":
            self._current_snippet.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._mode == "title":
            self._mode = None
        # Keep snippet mode across nested markup. It is flushed when the next
        # result title starts, which avoids truncating snippets at <b>/<a>/<div>.

    def close(self):
        super().close()
        self._flush()

    def _flush(self):
        if not self._current_url:
            return
        title = html.unescape(" ".join("".join(self._current_title).split())).strip()
        snippet = html.unescape(" ".join("".join(self._current_snippet).split())).strip()
        if title and self._current_url.startswith(("http://", "https://")):
            self._rank += 1
            domain = urlparse(self._current_url).netloc.lower()
            self.results.append(ResearchSource(title=title, url=self._current_url, snippet=snippet, source_domain=domain, rank=self._rank))
        self._current_url = ""
        self._current_title = []
        self._current_snippet = []
        self._mode = None

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not url:
            return ""
        # DDG result links can be redirect URLs containing uddg=<target>.
        parsed = urlparse(html.unescape(url))
        query = parse_qs(parsed.query)
        target = query.get("uddg", [""])[0]
        if target:
            return unquote(target)
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("/"):
            return urljoin("https://duckduckgo.com", url)
        return url


@dataclass
class DuckDuckGoSearchProvider(SearchProvider):
    endpoint: str = "https://html.duckduckgo.com/html/"
    user_agent: str = "PhoenixAI/1.0 (+local research agent)"
    name: str = "duckduckgo"

    async def search(self, request: ResearchRequest) -> list[ResearchSource]:
        query = request.query.strip()
        if not query:
            return []
        params = {"q": query, "kp": "-2", "kl": "us-en"}
        timeout = max(2.0, float(request.timeout_seconds))
        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(self.endpoint, params=params)
            response.raise_for_status()
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        parser.close()
        return _dedupe_sources(parser.results)[: max(1, request.max_sources)]


def _dedupe_sources(sources: list[ResearchSource]) -> list[ResearchSource]:
    seen: set[str] = set()
    result: list[ResearchSource] = []
    for source in sources:
        key = source.url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result
