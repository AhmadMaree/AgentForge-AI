import math

from datetime import datetime, timezone


from agentforge.memory.base import MemoryBase , MemoryConfig, ItemMemory
from agentforge.memory.storage.document_store import SQLiteDecumentStore
from agentforge.memory.storage.qdrant_store import QDrantVectorStore
from agentforge.memory.embedding import BaseEmbedding



class EpisodicMemory(MemoryBase):

    def __init__(
        self,
        config:MemoryConfig,
        user_id: str,
        document_store:SQLiteDecumentStore,
        vector_store: QDrantVectorStore,
        embedder: BaseEmbedding,
    ):
        super().__init__(config)

        if not user_id:
            raise ValueError("user id can't be empty")

        if vector_store.vector_size != embedder.dimension:
            raise ValueError("Vector store and embedder dimensions differ")
        
        self.user_id = user_id
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedder = embedder


    def add(self, item: ItemMemory) -> str:

        if item.user_id != self.user_id:
            raise ValueError("This ID belonges to anther memory")
        if item.memory_type != "episodic":
            raise ValueError("We accept episodic memory")

        #Generate vector from embedding
        vector = self.embedder.embed(
            item.content
        )

        #SQLite
        self.document_store.save(item)

        #if fail rebuild index.

        self.vector_store.add(
            meomery_id=item.id,
            vector=vector,
            user_id=item.user_id
        )

        return item.id

    def retrieve(self, query:str, limit:int = 5) -> list[ItemMemory]:

        if limit <= 0 :
            raise ValueError("limit should be positive")

        if not query.strip():
            return []

        query_vector = self.embedder.embed(query)

        hits = self.vector_store.serach(
            vector=query_vector,
            user_id=self.user_id,
            limit=limit * 5
        )

        ranked = []

        for memory_id, score in hits:
            item = self.document_store.get(
                memory_id=memory_id,
                user_id=self.user_id
            )

            if (item is None or item.memory_type != "episodic"):
                continue

            age_days = max(
                0.0,
                (
                    datetime.now(timezone.utc) - item.timestamp
                ).total_seconds() / 86400
            )
            recency = math.exp(
                -0.05 * age_days
            )

            similarity = max(
                0.0,
                min(1.0, score)
            )

            base_score = (
                0.8 * similarity + 0.2 * recency
            )

            final_score = (
                base_score * (0.8 + 0.4 * item.importance)
            )

            ranked.append(
                (final_score, item)
            )

        ranked.sort(
            key=lambda x : x[0],
            reverse=True
        )

        return [
            item
            for _,item in ranked[:limit]
        ]

    def list_by_session(
        self,
        session_id: str,
    ) -> list[ItemMemory]:

        return self.document_store.list_memories(
            user_id=self.user_id,
            memory_type ="episodic",
            session_id=session_id
        )


    def remove(
        self,
        memory_id: str,
    ) -> bool:
        item = self.document_store.get(
            memory_id=memory_id,
            user_id=self.user_id
        )

        if (
            item is None or item.memory_type != "episodic"
        ):
            return False

        deleted = self.document_store.delete(
            memory_id=memory_id,
            user_id=self.user_id
        )

        if deleted :
            self.vector_store.remove(memory_id)

        return deleted

    def rebuild_index(self) -> int:

        items = self.document_store.list_memories(
            user_id=self.user_id,
            memory_type="episodic"
        )

        for item in items:
            self.vector_store.add(
                meomery_id=item.id,
                vector=self.embedder.embed(
                    item.content
                ),
                user_id=self.user_id
            )

        return len(items)
        

        
        
       