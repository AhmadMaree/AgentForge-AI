
import uuid
from datetime import datetime, timezone, timedelta
from agentforge.memory.base import (
    MemoryConfig,
    ItemMemory,
)
from agentforge.memory.types.working import (
    WorkingMemory,
)

from agentforge.memory.types.episodic import (
    EpisodicMemory
)

from agentforge.memory.types.semantic import (
    SemanticMemory,
)

class ManagerMemory:

    def __init__(
        self,
        user_id: str,
        config: MemoryConfig | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ):
        if not user_id:
            raise ValueError("User id is required")

        if(
            (episodic is not None and episodic.user_id != user_id) or
            (semantic is not None and semantic.user_id != user_id)
        ):
            raise ValueError("EpisodicMemory user_id mismatch")
        
        self.user_id = user_id

        self.config = (
            config
            if config is not None
            else MemoryConfig()
        )

        self.workingMemory = WorkingMemory(
            user_id = self.user_id,
            config = self.config
        )

        self.episodic = episodic
        self.semantic = semantic

    def add_memory(
        self,
        content: str,
        importance:int = 0.5,
        memory_type: str = "working",
        session_id: str = "default",
    ) -> str:
        if memory_type not in (
            "working",
            "episodic",
            "semantic",
        ):
            raise ValueError("Unsupported memory type")


        item = ItemMemory(
            user_id=self.user_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            metadata={
                "session_id": session_id
            }
        )

        if memory_type == "working":
            return self.workingMemory.add(item)

        if memory_type == "episodic":

            if self.episodic is None:
                raise RuntimeError(
                    "EpisodicMemory is not configured"
                )
            return self.episodic.add(item)

        if self.semantic is None:
            raise RuntimeError(
                "SemanticMemory is not configured"
            )

        return self.semantic.add(item)

    def retrieve_memories(
        self,
        query: str,
        limit: str = 5,
        memory_type: str | None = None
    ) -> list[ItemMemory]:
        if limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        if memory_type not in (
            None,
            "working",
            "episodic",
            "semantic"
        ):
            raise ValueError(
                "Unsupported memory type"
            )


        results = []

        if memory_type in (None, "working"):
            results.extend(self.workingMemory.retrieve(query=query,limit=limit))

        if (
            memory_type in (None, "episodic") 
            and self.episodic is not None
        ):
            results.extend(self.episodic.retrieve(
                query=query,
                limit=limit
            ))

        if(memory_type in (None, 'semantic')
            and self.semantic is not None
        ):
            results.extend(
                self.semantic.retrieve(
                    query=query,
                    limit=limit
                )
            )

        results.sort(
            key=lambda x: (
                x.importance,
                x.timestamp
            ),
            reverse=True
        )

        return results[:limit]

    def remove_memories(self, memories_id: str) -> bool:
        if self.workingMemory.remove(memories_id):
            return True
        if (self.episodic is not None and self.episodic.remove(memories_id)):
            return self.episodic.remove(memories_id)

        if self.semantic is not None:
            return self.semantic.remove(memories_id)
        return False
        
    def clear(self) -> None:
        self.workingMemory.clear()


    def consolidate_memories(
        self,
        importance_threshold: int = 0.7
    ) -> int:

        if self.episodic is None:
            raise RuntimeError("episodic not configure")

        if 0 <= importance_threshold <= 1:
            raise ValueError("Invalid threshold")

        consolidated = 0

        for item in list(self.workingMemory.memories):

            if(item.importance < importance_threshold):
                continue

            #take sanpshot
            episode = item.model_copy(
                update={
                    "id": str(uuid.uuid4()),
                    "memory_type":"episodic",
                    "metadata":{
                        **item.metadata,
                        "source_memory_id": item.id
                    }
                }
            )

            self.episodic.add(episode)
            self.workingMemory.remove(item.id)

            consolidated += 1

        return consolidated

    def forget_memories(
        self,
        strategy: str,
        threshold: int = 0.2,
        max_age_days: int = 30
    ) -> int:

        if self.episodic is None:
            raise RuntimeError(
                "EpisodicMemory is not configured"
            )
        
        if strategy not in (
            "importance_based",
            "time_based",
        ):
            raise ValueError(
                "Unsupported forgetting strategy"
            )

        if not 0 <= threshold <= 1:
            raise ValueError(
                "Invalid threshold"
            )

        if max_age_days < 0:
            raise ValueError(
                "max_age_days cannot be negative"
            )

        items = (
            self.episodic.document_store.list_memories(
                user_id=self.user_id,
                memory_type="episodic"
            )
        )

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        )

        removed = 0

        for item in items:
            should_forget = False

            if strategy == "importance_based":
                should_forget = (
                    item.importance < threshold
                )

            if strategy == "time_based":
                should_forget(
                    item.timestamp < cutoff
                )

            if should_forget:
                if self.episodic.remove(item.id):
                    removed += 1

        return removed

