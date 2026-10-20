
import asyncio
import os

from dotenv import load_dotenv

from agentforge.core.exceptions import ToolsError
from agentforge.tools.async_executor import (
    AsyncExcute,
    ToolTask,
)
from agentforge.tools.calculator import CalculatorTool
from agentforge.tools.chain import ToolChain
from agentforge.tools.registry import RegistryTool
from agentforge.tools.search import SearchTool


def create_registry() -> RegistryTool:
    load_dotenv()

    registry = RegistryTool()
    registry.registry(CalculatorTool())

    if os.getenv("TAVILY_API_KEY"):
        registry.registry(SearchTool())

    return registry


async def main():
    registry = create_registry()

    print("\n=== Registered Tools ===")
    print(registry.list_tools())

    print("\n=== Calculator ===")
    result = registry.execute_tool(
        "calculator",
        {"expression": "15 * 8 + 32"},
    )
    print(result)

    print("\n=== Tool Chain ===")

    chain = ToolChain(registry)

    chain.add_step(
        name="subtotal",
        tool_name="calculator",
        build_input=lambda initial, results: {
            "expression": initial["expression"]
        },
    )

    chain.add_step(
        name="total",
        tool_name="calculator",
        build_input=lambda initial, results: {
            "expression": (
                f"{results['subtotal']} + 20"
            )
        },
    )

    print(
        chain.run({
            "expression": "15 * 8 + 32"
        })
    )

    print("\n=== Concurrent Calculator Calls ===")

    executor = AsyncExcute(
        registry=registry,
        max_concurrency=3,
    )

    tasks = [
        ToolTask(
            tool_name="calculator",
            parameters={"expression": "10 + 20"},
        ),
        ToolTask(
            tool_name="calculator",
            parameters={"expression": "12 * 5"},
        ),
        ToolTask(
            tool_name="calculator",
            parameters={"expression": "100 / 4"},
        ),
    ]

    results = await executor.muliple_excute(tasks)

    for item in results:
        print(item)

    if "web_search" in registry.list_tools():
        print("\n=== Real Web Search ===")

        try:
            search_result = registry.execute_tool(
                "web_search",
                {
                    "query": (
                        "What is retrieval augmented "
                        "generation?"
                    ),
                    "max_results": 3,
                },
            )

            for page in search_result["results"]:
                print(
                    page["title"],
                    page["url"],
                )

        except ToolsError as exc:
            print(f"Search failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
