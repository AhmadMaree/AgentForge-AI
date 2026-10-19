
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentforge.core.config import LLMConfig
from agentforge.core.exceptions import LLMError
from agentforge.core.message import Message
from agentforge.agents.simple_agent import SimpleAgent
from agentforge.llm.gemini import GeminiLLM


@pytest.fixture
def fake_client():
    client = Mock()

    client.models.generate_content.return_value = (
        SimpleNamespace(text="Hello from Gemini!")
    )

    client.models.generate_content_stream.return_value = iter([
        SimpleNamespace(text="Hello"),
        SimpleNamespace(text=" "),
        SimpleNamespace(text="world"),
    ])

    return client


@pytest.fixture
def llm(fake_client):
    return GeminiLLM(
        config=LLMConfig(),
        api_key="",
        client=fake_client,
    )


def test_invoke(llm, fake_client):
    result = llm.invoke([
        Message(
            role="user",
            content="Hello",
        )
    ])

    assert result == "Hello from Gemini!"

    fake_client.models.generate_content.assert_called_once()


def test_stream(llm):
    chunks = list(
        llm.stream([
            Message(
                role="user",
                content="Hello",
            )
        ])
    )

    assert chunks == [
        "Hello",
        " ",
        "world",
    ]


def test_system_message_mapping(llm):
    contents, config = llm.prepare_request([
        Message(
            role="system",
            content="You are a teacher.",
        ),
        Message(
            role="user",
            content="Explain Python.",
        ),
    ])

    assert config.system_instruction == (
        "You are a teacher."
    )

    assert len(contents) == 1
    assert contents[0].role == "user"


def test_agent_passes_current_message_to_gemini(llm, fake_client):
    agent = SimpleAgent(
        name="PythonTutor",
        llm=llm,
        system_prompt="You are a teacher.",
    )

    agent.run("Explain Python.")

    request = fake_client.models.generate_content.call_args.kwargs
    assert request["config"].system_instruction == "You are a teacher."
    assert [
        part.text
        for content in request["contents"]
        for part in content.parts
    ] == ["Explain Python."]


def test_assistant_role_mapping(llm):
    contents, _ = llm.prepare_request([
        Message(
            role="user",
            content="Hi",
        ),
        Message(
            role="assistant",
            content="Hello",
        ),
        Message(
            role="user",
            content="How are you?",
        ),
    ])

    assert [
        content.role
        for content in contents
    ] == [
        "user",
        "model",
        "user",
    ]


def test_empty_messages(llm):
    with pytest.raises(ValueError):
        llm.invoke([])


def test_system_only_messages_are_rejected(llm):
    with pytest.raises(
        ValueError,
        match="A user or assistant message required",
    ):
        llm.prepare_request([
            Message(
                role="system",
                content="You are a teacher.",
            )
        ])


def test_tool_messages_not_supported_yet(llm):
    with pytest.raises(ValueError):
        llm.invoke([
            Message(
                role="tool",
                content="Tool result",
            ),
        ])


def test_api_error_is_wrapped(llm, fake_client):
    fake_client.models.generate_content.side_effect = (
        RuntimeError("Network unavailable")
    )

    with pytest.raises(LLMError) as error:
        llm.invoke([
            Message(
                role="user",
                content="Hello",
            )
        ])

    assert "Network unavailable" in str(error.value)


def test_stream_error_is_wrapped(llm, fake_client):
    fake_client.models.generate_content_stream.side_effect = (
        RuntimeError("Connection lost")
    )

    with pytest.raises(LLMError):
        list(
            llm.stream([
                Message(
                    role="user",
                    content="Hello",
                )
            ])
        )
