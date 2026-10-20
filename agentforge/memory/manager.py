
from agentforge.memory.base import (
    MemoryConfig,
    ItemMemory,
)
from agentforge.memory.types.working import (
    WorkingMemory,
)

class ManagerMemory:

    def __init__(
        self,
        user_id: str,
        config: MemoryConfig | None = None
    ):
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

    def add_memory(
        self,
        content: str,
        importance:int = 0.5
    ) -> str:

        item = ItemMemory(
            user_id=self.user_id,
            content=content,
            importance=importance,
            memory_type="working"
        )
        
        return self.workingMemory.add(item)

    def retrieve_memories(
        self,
        query: str,
        limit: str = 5
    ) -> list[ItemMemory]:

        return self.workingMemory.retrieve(query=query,limit=limit)

    def remove_memories(self, memories_id) -> bool:
        return self.workingMemory.remove(memory_id=memories_id)
        
    def clear(self) -> None:
        self.workingMemory.clear()