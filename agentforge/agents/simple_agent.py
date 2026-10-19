
from agentforge.core.agent import Agent, AgentConfig
from agentforge.llm.base import BaseLLM
from agentforge.core.history import ConversationHistory
from agentforge.core.message import Message
from agentforge.core.exceptions import LLMError

class SimpleAgent(Agent):

    def __init__(
        self, 
        name: str, 
        llm: BaseLLM, 
        system_prompt : str = "", 
        config: AgentConfig | None = None,
        history: ConversationHistory | None = None
    ):

        super().__init__(
            name= name,
            llm=llm,
            system_prompt=system_prompt,
            config=config
        )

        self.history = (
            history
            if history is not None
            else ConversationHistory(
                max_maessages= self.config.max_history_messages
            )
        )

    def run(self, input_text: str) -> str:
        if not input_text.strip():
            raise ValueError("User input can't be empty")

        user_msg = Message(
            role="user",
            content=input_text
        )

        messages = self.history.build_context(
            current_user_message=user_msg,
            system_prompt=self.system_prompt
        )

        if self.config.debuge:
            print(
                f'[DEBUG]: Sending {len(messages)} to LLM'
            )

        response = self.llm.invoke(messages)

        if not response or not response.strip():
            raise LLMError("The LLM return an  empty answer")

        assistant_msg= (
            Message(
                role="assistant",
                content=response
            )
        )

        self.history.add_turn(
            user_massege=user_msg,
            assistant_message=assistant_msg
        )

        return response

    def reset(self) -> None:
        self.history.clear()

    def get_history(self) -> list[Message]:
        return self.history.get_messages()
       