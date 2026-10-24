
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from agentforge.rag.pipeline import RAGPipeline
from agentforge.tools.base import Tool


class RAGSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    enable_mqe: bool = False
    enable_hyde: bool = False


class RAGTool(Tool):
    name: ClassVar[str] = "rag_search"

    description: ClassVar[str] = (
        "Search the authorized document knowledge base. "
        "Returns passages with source and page metadata."
    )

    input_schema: ClassVar[type[BaseModel]] = RAGSearchInput

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def _execute(self, parameters: RAGSearchInput) -> dict:
        hits = self.pipeline.retrieve(
            query=parameters.query,
            top_k=parameters.top_k,
            enable_mqe=parameters.enable_mqe,
            enable_hyde=parameters.enable_hyde,
        )

        return {
            "query": parameters.query,
            "results": [
                {
                    "citation": f"[{index}]",
                    "chunk_id": hit.chunk.id,
                    "content": hit.chunk.text,
                    "source": hit.chunk.source,
                    "page": hit.chunk.page,
                    "score": hit.score,
                }
                for index, hit in enumerate(hits, start=1)
            ],
        }
