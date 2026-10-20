
from dataclasses import dataclass
from typing import Any, Callable

from agentforge.tools.registry import RegistryTool


InputBuilder = Callable[
    [dict[str, Any], dict[str, Any]],
    dict[str, Any],
]


@dataclass(frozen=True)
class ChainStep:
    name: str
    tool_name: str
    build_input: InputBuilder


class ToolChain:
    """Sequential tool pipeline with shared results."""

    def __init__(
        self,
        registry: RegistryTool,
    ):
        self.registry = registry
        self.steps: list[ChainStep] = []

    def add_step(
        self,
        name: str,
        tool_name: str,
        build_input: InputBuilder,
    ) -> "ToolChain":
        if any(step.name == name for step in self.steps):
            raise ValueError(
                f"Duplicate chain step: {name}"
            )

        self.steps.append(
            ChainStep(
                name=name,
                tool_name=tool_name,
                build_input=build_input,
            )
        )

        return self

    def run(
        self,
        initial_input: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for step in self.steps:
            parameters = step.build_input(
                initial_input,
                results.copy(),
            )

            result = self.registry.execute_tool(
                step.tool_name,
                parameters,
            )

            results[step.name] = result

        return results
