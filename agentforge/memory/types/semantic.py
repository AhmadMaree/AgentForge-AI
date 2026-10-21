import re

from agentforge.memory.base import (
    MemoryBase,
    MemoryConfig,
    ItemMemory,
)
from agentforge.memory.embedding import BaseEmbedding
from agentforge.memory.extraction import KnowledgeExtractor
from agentforge.memory.fusion import reciprocal_rank_fusion
from agentforge.memory.graph_models import (
    ExtractedKnowledge,
    GraphFact,
    normalize_name,
)
from agentforge.memory.storage.document_store import (
    SQLiteDecumentStore,
)
from agentforge.memory.storage.neo4j_store import (
    Neo4jGraphStore,
)
from agentforge.memory.storage.qdrant_store import (
    QDrantVectorStore,
)


class SemanticMemory(MemoryBase):
    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        documents: SQLiteDecumentStore,
        vectors: QDrantVectorStore,
        graph: Neo4jGraphStore,
        embedder: BaseEmbedding,
        extractor: KnowledgeExtractor,
    ):
        super().__init__(config)

        if not user_id:
            raise ValueError("user_id cannot be empty")

        if vectors.vector_size != embedder.dimension:
            raise ValueError("Embedding dimension mismatch")

        self.user_id = user_id
        self.documents = documents
        self.vectors = vectors
        self.graph = graph
        self.embedder = embedder
        self.extractor = extractor

    def add(self, item: ItemMemory) -> str:
        if item.user_id != self.user_id:
            raise ValueError("Wrong memory owner")

        if item.memory_type != "semantic":
            raise ValueError("Expected semantic memory")

        # Prepare all derived data first.
        knowledge = self.extractor.extract(item.content)
        vector = self.embedder.embed(item.content)

        # Store the extracted graph data in SQLite
        # so failed indexes can be rebuilt later.
        persisted = item.model_copy(
            deep=True,
            update={
                "metadata": {
                    **item.metadata,
                    "extracted_knowledge": (
                        knowledge.model_dump()
                    ),
                }
            },
        )

        self.documents.save(persisted)

        # Secondary indexes: these operations are
        # not part of the SQLite transaction.
        self.vectors.add(
            meomery_id=item.id,
            vector=vector,
            user_id=self.user_id,
        )

        self.graph.add_knowledge(
            user_id=self.user_id,
            memory_id=item.id,
            entities=knowledge.entities,
            relations=knowledge.relations,
        )

        return item.id

    def _query_entities(self, query: str) -> list[str]:
        # Lightweight lexical entity matching.
        # Does not require another Gemini call.
        words = re.findall(
            r"\b[\w-]+\b",
            query,
        )

        return [
            word
            for word in words
            if len(word) >= 3
        ]

    def search_with_evidence(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[list[ItemMemory], list[GraphFact]]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        if not query.strip():
            return [], []

        vector_hits = self.vectors.serach(
            vector=self.embedder.embed(query),
            user_id=self.user_id,
            limit=limit * 5,
        )

        graph_facts = self.graph.search(
            user_id=self.user_id,
            names=self._query_entities(query),
            limit=limit * 10,
        )

        vector_ids = [
            memory_id
            for memory_id, _ in vector_hits
        ]

        # Preserve first occurrence of each memory.
        graph_ids = list(dict.fromkeys(
            fact.memory_id
            for fact in graph_facts
        ))

        fused = reciprocal_rank_fusion([
            vector_ids,
            graph_ids,
        ])

        memories = []

        for memory_id, _score in fused:
            item = self.documents.get(
                memory_id=memory_id,
                user_id=self.user_id,
            )

            if (
                item is not None
                and item.memory_type == "semantic"
            ):
                memories.append(item)

            if len(memories) >= limit:
                break

        selected_ids = {
            item.id for item in memories
        }

        selected_facts = [
            fact
            for fact in graph_facts
            if fact.memory_id in selected_ids
        ]

        return memories, selected_facts

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[ItemMemory]:
        memories, _ = self.search_with_evidence(
            query,
            limit,
        )
        return memories

    def remove(self, memory_id: str) -> bool:
        item = self.documents.get(
            memory_id,
            self.user_id,
        )

        if (
            item is None
            or item.memory_type != "semantic"
        ):
            return False

        deleted = self.documents.delete(
            memory_id,
            self.user_id,
        )

        if deleted:
            self.vectors.remove(memory_id)
            self.graph.remove_memory(
                self.user_id,
                memory_id,
            )

        return deleted

    def rebuild_indexes(self) -> int:
        items = self.documents.list_memories(
            user_id=self.user_id,
            memory_type="semantic",
        )

        for item in items:
            raw = item.metadata.get(
                "extracted_knowledge",
                {},
            )

            knowledge = (
                ExtractedKnowledge.model_validate(raw)
            )

            self.vectors.add(
                meomery_id=item.id,
                vector=self.embedder.embed(
                    item.content
                ),
                user_id=self.user_id,
            )

            self.graph.add_knowledge(
                user_id=self.user_id,
                memory_id=item.id,
                entities=knowledge.entities,
                relations=knowledge.relations,
            )

        return len(items)