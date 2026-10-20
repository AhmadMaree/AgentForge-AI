
from typing import Any
from agentforge.tools.base import Tool
from agentforge.core.exceptions import (
    ToolsError,
    ToolNotFoundError
)

class RegistryTool:
    def __init__(self):
        self._tools: dict[str,Tool] = {}

    def registry(self, tool: Tool) -> None:

        if not isinstance(tool, Tool):
            raise TypeError("only tool can be registry")

        if not tool and not tool.name.strip():
            raise ValueError("Tool name cannot be empty.")

        if tool.name in self._tools:
            raise ToolsError("Tools already registry")

        self._tools[tool.name] = tool

    def unregistry(self, name: str) -> bool:
        return self._tools.pop(name,None) is not None

    def getTool(self, name:str) -> Tool:
        tool = self._tools.get(name)

        if not tool:
            raise ToolNotFoundError(
                f"Tool {name} not registry"
            )

        return tool

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_tools_description(self) -> str:

        descriptions = []

        for tool in self._tools.values():
            descriptions.append(
                f"{tool.name}: {tool.description}"
            )
        return "\n\n".join(descriptions)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema.model_json_schema()
            }
            for tool in self._tools.values()
        ]

    def execute_tool(
        self,
        name: str,
        parameters: dict[str, Any],
    ) -> Any:
        tool = self.getTool(name)
        return tool.run(parameters)

    async def aexecute_tool(
        self,
        name: str,
        parameters: dict[str, Any],
    ) -> Any:
        tool = self.getTool(name)
        return await tool.arun(parameters)

