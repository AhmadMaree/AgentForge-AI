
from agentforge.memory.fusion import reciprocal_rank_fusion
from agentforge.rag.chunker import TextChunker
from agentforge.rag.expansion import QueryExpander
from agentforge.rag.lexical import BM25Retriever
from agentforge.rag.loaders import load_document
from agentforge.rag.models import Chunk, SearchHit
from agentforge.rag.reranker import OverlapReranker
from agentforge.rag.store import RAGVectorStore


class RAGPipeline:
    def __init__(
        self,
        store: RAGVectorStore,
        user_id: str,
        namespace: str,
        chunker: TextChunker | None = None,
        expander: QueryExpander | None = None,
        reranker=None,
    ):
        if not user_id.strip() or not namespace.strip():
            raise ValueError(
                "user_id and namespace are required"
            )

        self.store = store
        self.user_id = user_id
        self.namespace = namespace
        self.chunker = chunker or TextChunker()
        self.expander = expander
        self.reranker = reranker or OverlapReranker()

        self.chunks: list[Chunk] = []
        self.refresh()

    def refresh(self) -> None:
        # Rebuild the lexical index from stored chunks.
        self.chunks = self.store.list_chunks(
            user_id=self.user_id,
            namespace=self.namespace,
        )

        self.lexical = BM25Retriever(self.chunks)

    def ingest(self, path: str) -> int:
        # Loading and indexing happen outside RAGTool.
        documents = load_document(path)
        chunks = self.chunker.spilt(documents)

        count = self.store.add(
            chunks=chunks,
            user_id=self.user_id,
            namespace=self.namespace,
        )

        self.refresh()
        return count

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
    ) -> list[SearchHit]:
        if not query.strip():
            return []

        if top_k <= 0 or candidate_pool_multiplier <= 0:
            raise ValueError("Retrieval limits must be positive")

        if (enable_mqe or enable_hyde) and self.expander is None:
            raise RuntimeError(
                "MQE/HyDE require a configured QueryExpander"
            )

        pool_size = max(
            top_k * candidate_pool_multiplier,
            20,
        )

        dense_queries = [query]

        if enable_mqe:
            dense_queries.extend(
                self.expander.multi_query(query)
            )

        if enable_hyde:
            hypothetical = self.expander.hyde(query)

            if hypothetical:
                dense_queries.append(hypothetical)

        dense_queries = list(
            dict.fromkeys(dense_queries)
        )

        rankings: list[list[str]] = []
        candidates: dict[str, SearchHit] = {}

        # Dense rankings: original question and optional expansions.
        for search_query in dense_queries:
            hits = self.store.search(
                query=search_query,
                user_id=self.user_id,
                namespace=self.namespace,
                limit=pool_size,
            )

            rankings.append([
                hit.chunk.id for hit in hits
            ])

            for hit in hits:
                candidates.setdefault(hit.chunk.id, hit)

        # BM25 uses the original query to preserve exact terms.
        lexical_hits = self.lexical.search(
            query=query,
            limit=pool_size,
        )

        rankings.append([
            hit.chunk.id for hit in lexical_hits
        ])

        for hit in lexical_hits:
            candidates.setdefault(hit.chunk.id, hit)

        fused = reciprocal_rank_fusion(rankings)

        fused_hits = []

        for chunk_id, fused_score in fused[:pool_size]:
            hit = candidates.get(chunk_id)

            if hit is not None:
                fused_hits.append(
                    SearchHit(
                        chunk=hit.chunk,
                        score=float(fused_score),
                        method="hybrid",
                    )
                )

        return self.reranker.rerank(
            query=query,
            hits=fused_hits,
            limit=top_k,
        )
