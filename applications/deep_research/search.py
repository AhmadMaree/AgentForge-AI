
import os
from .models import (
    SearchResult
)

from urllib.parse import urlsplit, urlunsplit
from typing import Protocol

class SearchProvider(Protocol):
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        ...


def normalize_url(url: str) -> str:
        parts = urlsplit(url=url.strip())

        if parts.scheme not in ("https","http"):
            raise ValueError("Unsupported URL scheme")

        if not parts.hostname:
            raise ValueError("Missing URL hostname")

        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/") or "/",
                parts.query,
                "",
            )
        )


def duplicates_source(res: list[SearchResult]) -> list[SearchResult]:
        seen = set()
        uniqe = []

        for source in res:
            key = normalize_url(str(source.url))

            if key in seen:
                 continue

            seen.add(key)
            uniqe.append(source)

        return uniqe


class DuckDuckGoSearchProvider:

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:

        try:
            from ddgs import DDGS
        except ImportError as err:
            raise RuntimeError(
                "Install ddgs to use DuckDuckGo search."
            ) from err

        with DDGS() as client:
            list_results = list(
                client.text(
                    query,
                    max_results = max_results
                )
            )

        res = []


        for item in list_results:

            url = item.get("href").strip() or item.get("url")
            title = item.get("title") or url

            if not url:
                continue
            try:

                res.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=(
                            item.get("body") or ""
                        )[:1500],
                        backend="duckduckgo"
                    )
                )
            except ValueError:
                continue

        return duplicates_source(res)




class TavilySearchProvider:

    def __init__(self):
        from  agentforge.tools.search import SearchTool

        self.tool = SearchTool()


    def search(
        self,
        query: str,
        max_results: int = 3
    ) -> list[SearchResult]:

        from agentforge.tools.search import SearchInput

        response = self.tool.run(
            {
                "query": query,
                "max_results": max_results,
                "search_depth": "basic"
            }
        )

        if isinstance(response, str):
            import json
            response = json.loads(response)

        if not isinstance(response, dict):
            raise TypeError(
                "Unexpected Tavily search response"
            )

        results = []


        for item in response.get("results", []):

            if not item.get["url"]:
                continue

            results.append(
                SearchResult(
                    title=item.get("title") or item["url"],
                    url=item["url"],
                    snippet=(
                        item.get("content") or ""
                    )[:2000],
                    backend="tavily",
                )
            )

        return duplicates_source(results)




class SearchService:

    def __init__(self, provider: SearchProvider | None = None):

        if provider is not None:
            self.provider = provider
            return
        
        backend = os.getenv(
            "SEARCH_BACKEND", "duckduckgo"
        ).lower()

        if backend == "duckduckgo":
            self.provider = DuckDuckGoSearchProvider()

        elif backend == "tavily":
            self.provider = TavilySearchProvider()
        else:
            raise ValueError(
                f"Unknown search backend: {backend}"
            )


    def search(
        self,
        query: str,
        max_results: int = 5,
    )-> list[SearchResult]:

        if not query.strip():
            raise ValueError("Search query is empty")

        results = self.provider.search(
            query, max_results
        )

        return duplicates_source(results)[:max_results]

        


         








        