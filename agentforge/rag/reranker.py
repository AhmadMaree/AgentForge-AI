from agentforge.rag.lexical import tokenize
from agentforge.rag.models import SearchHit



class OverlapReranker:
    
    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        limit: int = 5,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_terms = set(tokenize(query))\

        def relevance(hit: SearchHit) -> float:
            passage_terms = set(
                tokenize(hit.chunk.text)
            )

            if not query_terms:
                return 0.0

            return (
                len(query_terms & passage_terms)
                / len(query_terms)
            )

        return sorted(
            hits,
            key=lambda hit: (
                -relevance(hit),
                -hit.score,
                hit.chunk.id,
            ),
        )[:limit]

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        limit: int = 5,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        if not hits:
            return []

        pairs = [
            (query, hit.chunk.text)
            for hit in hits
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, hits),
            key=lambda pair: float(pair[0]),
            reverse=True,
        )

        # Preserve each original hit's metadata.
        return [
            hit for _, hit in ranked[:limit]
        ]    