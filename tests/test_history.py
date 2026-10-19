from agentforge.core.history import ConversationHistory
from agentforge.core.message import Message


def test_build_context_includes_current_user_message():
    history = ConversationHistory(max_maessages=4)
    previous_user_message = Message(role="user", content="Earlier question")
    previous_assistant_message = Message(
        role="assistant",
        content="Earlier answer",
    )
    current_user_message = Message(role="user", content="Current question")
    history.add_turn(previous_user_message, previous_assistant_message)

    messages = history.build_context(
        current_user_message=current_user_message,
        system_prompt="You are a teacher.",
    )

    assert [
        (message.role, message.content)
        for message in messages
    ] == [
        ("system", "You are a teacher."),
        ("user", "Earlier question"),
        ("assistant", "Earlier answer"),
        ("user", "Current question"),
    ]
