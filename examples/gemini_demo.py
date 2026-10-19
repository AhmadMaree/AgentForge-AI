
from agentforge.core.message import Message
from agentforge.llm.factory import FactoryLLM

def main():
    llm = FactoryLLM.create("gemini")

    messages = [
        Message(
            role="system",
            content= (
                "You are an expert AI engineering teacher. "
                "Explain concepts clearly and provide examples."
            )
        ),
        Message(
            role="user",
            content=(
                "What is the difference between "
                "an LLM and an AI agent?"
            ),
        )
    ]

    answer = llm.invoke(messages)

    print(messages)

    print("\n=== Streaming Generation ===\n")

    streaming_messages = [
        Message(
            role="user",
            content=(
                "Explain how the ReAct agent pattern "
                "works, with a Python example."
            ),
        ),
    ]

    for chunk in llm.stream(messages=streaming_messages):
        print(chunk, end="", flush=True)

    print()

if __name__ == "__main__":
    main()
