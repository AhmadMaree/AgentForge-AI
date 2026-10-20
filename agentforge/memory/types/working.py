
from datetime import datetime, timezone , timedelta

from agentforge.memory.base import MemoryBase, MemoryConfig, ItemMemory

from agentforge.utils.utils import toknizer

class WorkingMemory(MemoryBase):

    def __init__(self, config: MemoryConfig, user_id: str):
        super().__init__(config)

        if not user_id:
            raise ValueError("User ID can't be empty")
        
        self.user_id = user_id
        self.memories: list[ItemMemory] = []

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _expire_old_memories(self) -> None:

        cutoff = self._now() - timedelta(
            minutes=self.config.working_memory_ttl
        )

        self.memories = [
            item
            for item in self.memories
            if item.timestamp > cutoff
        ]

    def _remove_lowest_proirity_memory(
        self
    ) -> None:
        if not self.memories:
            return

        #Here we looking intp lowest importance if equal look into timestamp.
        lowest = min(
            self.memories,
            key=lambda x: (
                x.importance,
                x.timestamp
            )
        )

        self.memories.remove(lowest)

    def add(self, item: ItemMemory) -> str:

        if item.user_id != self.user_id:
            raise ValueError("This ID belonges to anther memory")
        if item.memory_type != "working":
            raise ValueError("We accept only working memory")

        self._expire_old_memories()
        cutoff = self._now() - timedelta(
            minutes=self.config.working_memory_ttl
        )
        if item.timestamp <= cutoff:
            raise ValueError("Can't add expired memory")

        if any(
            item.id == existing.id
            for existing in self.memories
        ):
            raise ValueError("Memory ID already exists")

        if self.memories >= self.config.working_memory_capactiy:
            self._remove_lowest_proirity_memory()

        self.memories.append(item)
        return item.id

    def retrieve(self, query: str, limit: int = 5) -> list[ItemMemory]:

        if limit <=0 :
            raise ValueError("Limit should be positive")

        self._expire_old_memories()

        query_token = toknizer(query)

        if not query_token:
            return []

        scored = list[tuple[float,ItemMemory]] = []

        for item in self.memories:
            content_token = toknizer(item.content)

            if not content_token:
                continue

            overlap = len(content_token & query_token)
            if overlap == 0:
                continue

            relevance = (overlap / len(query_token))
            importance_wight = item.importance * 0.4 + 0.8

            age_min = max(0.0, (self._now() - item.timestamp) / 60)
            time_deacy = 1 / (1 + age_min / 60)
            score = (relevance * importance_wight * time_deacy)

            scored.append((score, item))

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            item
            for _,item in scored[:limit]
        ]

    def remove(self, memory_id: str) -> bool:
        self._expire_old_memories()

        for index, item in enumerate(
            self.memories
        ):
            if item.id == memory_id:
                del self.memories[index]
                return True
        return False

    def clear(self) -> None:
        self.memories.clear()

    def count(self) -> int:
        self._expire_old_memories()
        return len(self.memories)



