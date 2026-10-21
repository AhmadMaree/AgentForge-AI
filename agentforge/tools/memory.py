
from typing import ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agentforge.memory.manager import ManagerMemory
from agentforge.tools.base import Tool


class MemorySearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=500,
    )

    memory_type: Literal[
        "working",
        "episodic",
        "semantic",
    ] = "semantic"

    limit: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class MemoryTool(Tool):
    name: ClassVar[str] = "memory_search"

    description: ClassVar[str] = (
        "Search authorized stored memories "
        "for relevant facts and experiences. "
        "Returns memory content and source IDs."
    )

    input_schema: ClassVar[type[BaseModel]] = (
        MemorySearchInput
    )

    def __init__(self, manager: ManagerMemory):
        self.manager = manager

    def _excute(
        self,
        parameters: MemorySearchInput,
    ) -> dict:
        if (
            parameters.memory_type == "semantic"
            and self.manager.semantic is not None
        ):
            memories, facts = (
                self.manager.semantic.search_with_evidence(
                    query=parameters.query,
                    limit=parameters.limit,
                )
            )

            return {
                "memories": [
                    {
                        "id": item.id,
                        "content": item.content,
                        "importance": item.importance,
                    }
                    for item in memories
                ],
                "relationships": [
                    fact.model_dump()
                    for fact in facts
                ],
            }

        memories = self.manager.retrieve_memories(
            query=parameters.query,
            limit=parameters.limit,
            memory_type=parameters.memory_type,
        )

        return {
            "memories": [
                {
                    "id": item.id,
                    "content": item.content,
                    "importance": item.importance,
                }
                for item in memories
            ],
            "relationships": [],
        }
