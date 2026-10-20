
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from agentforge.llm.base import BaseLLM

class AgentConfig(BaseModel):
    max_history_messages: int = Field(
        default=10,
        ge=2
    )

    debuge: bool = False

class Agent(ABC):

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: str = "",
        config: AgentConfig | None = None
    ):
        if not name:
            raise ValueError("Agent name is required")

        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.config = config or AgentConfig()

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Excute agent task"""
        ...