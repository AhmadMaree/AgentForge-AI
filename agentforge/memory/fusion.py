from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("k must be positive")

    scores = defaultdict(float)

    for ranking in rankings:
        seen = set()

        for position, document_id in enumerate(
            ranking,
            start=1,
        ):
            if document_id in seen:
                continue

            seen.add(document_id)

            scores[document_id] += (
                1.0 / (k + position)
            )

    return sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )