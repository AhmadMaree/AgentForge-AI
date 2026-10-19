from agentforge.llm.factory import FactoryLLM
from agentforge.core.agent import AgentConfig
from agentforge.agents.simple_agent import SimpleAgent

from agentforge.core.exceptions import (
    ConfigurationError,
    LLMError
)

HELP_TEXT = """
Commands:
  /help     Show available commands
  /history  Display conversation history
  /reset    Clear conversation history
  /exit     Quit the application
"""

def main():

    print("=" * 55)
    print("AgentForge - CLI (Gemini Chat)")
    print("=" * 55)

    try:
        llm = FactoryLLM.create("gemini")
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return

    agent = SimpleAgent(
        name="PythonTutor",
        llm=llm,
        system_prompt=(
            "You are an expert Python and AI engineering "
            "teacher. Explain concepts step by step. "
            "Provide clear examples and help the user "
            "understand why the code works."
        ),
        config=AgentConfig(
            max_history_messages=20,
            debuge=False
        )
    )

    print(HELP_TEXT)

    while True:

        try:
            user_input = input("\nYou:").strip()
        except (KeyboardInterrupt, EOFError) as err:
            print("Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "/exit":
            print("Goodbye!")
            break

        if cmd == "/help":
            print(HELP_TEXT)
            continue

        if cmd == "/reset":
            agent.reset()
            print("Conversation history cleared.")
            continue

        if cmd == "/history":
            history = agent.get_history()

            if not history:
                print("No convestion history")
                continue

            for msg in history:
                print(
                    f"\n{msg.role.upper()}"
                    f"{msg.content}"
                )

            continue

        if cmd.startswith('/'):
            print("Unkown cmd type /help")
            continue

        try:
            res = agent.run(user_input)
            print(
                f"\n{agent.name}: {res}"
            )
        except LLMError as err:
            print(f"\n Modal erro : {err}")
        except ValueError as exc:
            print(f"\n input error: {exc}")


if __name__ == "__main__":
    main()
