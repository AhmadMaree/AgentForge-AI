
import asyncio

from dataclasses import dataclass
from typing import Any

from agentforge.tools.registry import RegistryTool
from agentforge.core.exceptions import (
    ToolsError
)

@dataclass(frozen=True)
class ToolTask:
   tool_name: str
   parameters: dict[str, Any]

@dataclass(frozen=True)
class ToolTaskResults:
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None



class AsyncExcute:

    def __init__(
        self,
        registry:RegistryTool,
        max_concurrency: int = 3
    ):
        if max_concurrency < 1:
            raise ValueError(
                "max_concurrency must be >= 1"
            )

        self.registry = registry
        self.semphore = asyncio.Semaphore(
            max_concurrency
        )

    async def excute_task(
        self,
        task: ToolTask
    ) -> ToolTaskResults:

        async with self.semphore:
            try:
                result = await self.registry.aexecute_tool(name=task.tool_name, parameters=task.parameters)
                return ToolTaskResults(
                    tool_name= task.tool_name,
                    result= result,
                    success=True
                )
            except ToolsError as err:
                return ToolTaskResults(
                    tool_name=task.tool_name,
                    success=False,
                    error= str(err)
                )

    async def muliple_excute(
        self,
        tasks: list[ToolTask]
    )-> list[ToolTaskResults]:
        return await asyncio.gather(
            *[
                self.excute_task(task)
                for task in tasks
            ]
        )
    
        
        
        
