
import uuid
from abc import ABC , abstractmethod
from pydantic import Field, BaseModel
from datetime import datetime, timezone
from typing import Literal, Any

#Types of memory
MemoryType = Literal[
    "working",
    "episodic",
    "semantic",
    "perceptual"
]

def utc_now() -> datetime :
    return datetime.now(timezone.utc)


class MemoryConfig:
    working_memory_capactiy: int = Field(
        default=50,
        gt= 0
    )

    working_memory_ttl: int = Field(
        default=60,
        gt=0,
        description="TTL in minutes"
    )

class ItemMemory(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    user_id: str = Field(max_length=1)
    content: str = Field(max_length=1)
    timestamp: datetime = Field(
        default_factory=utc_now
    )

    memory_type: MemoryType = "working"

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )

class MemoryBase(ABC):

    def __init__(self, config:MemoryConfig):
       self.config = config

    @abstractmethod
    def add(self, item: ItemMemory):
        """Add to memory"""
    
    @abstractmethod
    def retrieve(self, query: str, limit:int = 5) -> list[ItemMemory]:
        """Retreive revelant memorize"""

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """remove a memory by ID"""