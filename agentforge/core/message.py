
from pydantic import BaseModel, Field
from typing import Literal, Any

from datetime import datetime, timezone


MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool"
]

class Message(BaseModel):
    role: MessageRole
    content: str

    #timestamp
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

    def to_dict(self) -> dict[str,str]:
        return {
            "role": self.role,
            "content": self.content
        }