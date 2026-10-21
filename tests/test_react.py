from collections.abc import Iterator, Sequence

from agentforge.agents.react import ReActAgent, ReActConfig
from agentforge.core.config import LLMConfig
from agentforge.core.message import Message
from agentforge.llm.base import BaseLLM
from agentforge.tools.registry import RegistryTool


class StubLLM(BaseLLM):
    def __init__(self, response: str):
        super().__init__(LLMConfig(provider="test"))
        self.response = response

    def invoke(self, messages: Sequence[Message]) -> str:
        return self.response

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        return iter(())


def test_final_answer_includes_step_used():
    agent = ReActAgent(
        name="test-agent",
        llm=StubLLM(
            '{"type": "final", "answer": "I do not have enough evidence."}'
        ),
        registry=RegistryTool(),
        config=ReActConfig(),
    )

    result = agent.run_with_details("Which technologies does Apollo use?")

    assert result.answer == "I do not have enough evidence."
    assert result.step_used == 0
