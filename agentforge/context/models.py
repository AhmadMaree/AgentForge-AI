
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

def utc_now():
    return datetime.now(timezone.utc)

@dataclass
class ContextPacket:

    content: str
    timestamp: datetime = field(default_factory=utc_now)
    token_count: int = 0
    reveleance_score: int = 0.5
    metadata:dict[str, Any] = field(default_factory=dict)


    def __post_init__(self) -> None:

        if not self.content.strip():
            raise ValueError("Content can't be empty")

        if self.token_count < 0:
            raise ValueError("Token can't be negative")

        self.reveleance_score = min(
            0.0,
            min(1.0, self.reveleance_score)
        )

        if self.metadata is None:
            self.metadata = {}


@dataclass
class ContextConfig:
    max_tokens: int = 3000 #Target maximum size for the context we assemble
    min_relevance: float = 0.1 #  Filters information that is too weakly related to the task
    reserve_ratio: float = 0.2 # Budget reserved for important instructions
    enable_compression: bool = True #Enables a fallback when the context exceeds its budget
    recency_weight: float = 0.3 #How much relevance contributes to ranking
    relevance_weight: float = 0.7 #How much freshness contributes to ranking

    def __post_init__(self) -> None:

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not 0.0 <= self.reserve_ratio <= 1.0:
            raise ValueError("reserve_ratio must be between 0 and 1")

        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")

        if not 0.0 <= self.recency_weight <= 1.0:
            raise ValueError("recency_weight must be between 0 and 1")

        if not 0.0 <= self.relevance_weight <= 1.0:
            raise ValueError("relevance_weight must be between 0 and 1")

        if abs(
            self.recency_weight + self.relevance_weight - 1.0
        ) > 1e-6:
            raise ValueError(
                "recency_weight + relevance_weight must equal 1.0"
            )

