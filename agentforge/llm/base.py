
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from agentforge.core.config import LLMConfig
from agentforge.core.message import Message

class BaseLLM(ABC):

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def invoke(self, messages: Sequence[Message]) -> str:
        """Generate a complete response """
        ...

    @abstractmethod
    def stream(self, messages: Sequence[Message]) -> Iterator[str] :
        """Yelid response text incrementally."""
        ...