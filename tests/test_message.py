import pytest

from agentforge.core.message import Message
from pydantic import ValidationError

def test_create_user_message():
    message = Message(
       role="user",
       content="Hello AI Agent"
    )

    assert message.role == "user"
    assert message.content == "Hello AI Agent"

def test_convert_message_to_dict():
    message = Message(
        role="assistant",
        content="Test assitant message"
    )

    assert message.to_dict() == {"role":"assistant", "content":"Test assitant message"}


def test_invalid_role_meesage():

    with pytest.raises(ValidationError) :
        Message (
            role="admin",
            content="Hi!"
        )


def test_metadata_is_independent():
    first = Message(role="user", content="A")
    second = Message(role="user", content="B")

    first.metadata["source"] = "test"

    assert "source" not in second.metadata