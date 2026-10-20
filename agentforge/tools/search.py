
import os
from dotenv import load_dotenv

from agentforge.tools.base import Tool
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict , Field

from agentforge.core.exceptions import (
    ConfigurationError
)

from tavily import TavilyClient

class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(
       min_length=2,
       max_length=500,
       description="Web search query"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10
    )
    search_depth:str = Field(
        default="basic",
        pattern="^(basic|advanced)$"
    )


class SearchTool(Tool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Serach a publich web and return revelant pages, titles, Urls and snippets"
    input_schema: ClassVar[type[BaseModel]] = SearchInput

    def __init__(self, client:Any | None = None, api_key: str | None = None):

        load_dotenv()

        key = api_key or os.getenv("TAVILY_API_KEY")

        if client is None and not key:
            raise ConfigurationError("TAVILY_API_KEY is missing")

        self.client = (
            client
            if client is not None
            else TavilyClient(api_key=key)
        )

    def _excute(self, paramter: SearchInput) -> dict[str, Any]:

        response =  self.client.search(
            query=paramter.query,
            search_depth= paramter.search_depth,
            max_results= paramter.max_results,
            include_answer=False
        )

        results = []

        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url",""),
                "content": result.get("content", ""),
                "score": result.get("score", "")
            })

        return {
            "query": paramter.query,
            "results": results,
            "count": len(results)
        }


       

        