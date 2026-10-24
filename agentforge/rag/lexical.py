import math
import re

from collections import Counter

from agentforge.rag.models import Chunk, SearchHit


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b[\w-]+\b",
        text.casefold(),
    )

class BM25Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.frequencies = [
            Counter(tokenize(chunk.text))
            for chunk in chunks
        ]

        self.lengths = [
            sum(freq.values())
            for freq in self.frequencies
        ]

        self.avgdl = (
            sum(self.lengths) / len(self.lengths)
            if self.lengths else 0.0
        )

        self.df = Counter()

        for freq in self.frequencies:
            self.df.update(freq.keys())

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        if not self.chunks or not query.strip():
            return []

        query_terms = set(tokenize(query))
        total = len(self.chunks)
        results = []

        for chunk, freq, length in zip(
            self.chunks,
            self.frequencies,
            self.lengths,
        ):
            score = 0.0

            for term in query_terms:
                tf = freq.get(term, 0)

                if tf == 0:
                    continue

                df = self.df[term]

                idf = math.log(
                    1 + (
                        total - df + 0.5
                    ) / (df + 0.5)
                )

                denominator = tf + self.k1 * (
                    1 - self.b
                    + self.b * length / max(
                        self.avgdl, 1.0
                    )
                )

                score += (
                    idf * tf * (self.k1 + 1)
                    / denominator
                )

            if score > 0:
                results.append(
                    SearchHit(
                        chunk=chunk,
                        score=score,
                        method="bm25",
                    )
                )

        results.sort(
            key=lambda hit: (
                -hit.score,
                hit.chunk.id,
            )
        )

        return results[:limit]
