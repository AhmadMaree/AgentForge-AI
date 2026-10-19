
from collections import deque

from agentforge.core.message import Message

class ConversationHistory:

    def __init__(
        self,
        max_maessages: int = 20
    ):

        if max_maessages < 2:
            raise ValueError(
                "massages should be at least 2"
            )

        self.max_messages = max_maessages
        self.max_turns = max_maessages // 2

        self._turns : deque[tuple[Message, Message]] = deque(maxlen=self.max_turns)


    def add_turn(
        self,
        user_massege: Message,
        assistant_message: Message
    ) -> None :

        if user_massege.role != "user":
            raise ValueError(
                "First message must have user message"
            )

        if assistant_message.role != "assistant":
            raise ValueError("Second message must have role = 'assistant' ")

        self._turns.append(
            (user_massege,assistant_message)
        )


    def get_messages(self) -> list[Message]:

        messages = []

        for user_massege,assistant_message in self._turns:
            messages.extend([
                user_massege,
                assistant_message
            ])

        return messages


    def build_context(
        self,
        current_user_message: Message,
        system_prompt: str = ""
    ) -> list[Message]:
        if current_user_message.role != 'user':
            raise ValueError("Current message should be role = 'user'")

        messages = []

        if system_prompt:
            messages.append(
                Message(
                    role='system',
                    content=system_prompt
                )
            )

        pervious_count = self.max_turns - 1

        pervious_turns = (
            list(self._turns)[-pervious_count:]
            if pervious_count > 0
            else []
        )

        for user_msg, assistant_msg in pervious_turns:
            messages.extend([
                user_msg,
                assistant_msg
            ])

        messages.append(current_user_message)

        return messages


    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns) * 2
        