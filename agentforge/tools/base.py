
import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError
from agentforge.core.exceptions import (
    ToolValidtionError,
    ToolExcutionError,
    ToolsError
)

class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def _excute(self, paramter: BaseModel) -> Any:
        "Excute Tool (calling function)"
        ...
    
    def validtion_input(
        self,
        paramter: dict[str,Any]
    )-> BaseModel:
        try:
            return self.input_schema.model_validate(
                paramter
            )
        except ValidationError as err:
            raise ToolValidtionError(
                f"Invalid input for {self.name}"
                f"{err}"
            ) from err

    def run(
        self,
        paramter: dict[str, Any]
    ) -> Any:

        validtion = self.validtion_input(paramter)

        try:
            res = self._excute(validtion)
            return res
        except ToolsError:
            raise
        except Exception as exc:
            raise ToolExcutionError (
                f"Tool {self.name} faild: {exc}"
            ) from exc

    async def arun(
        self,
        param: dict[str, Any]
    ) -> Any :
        return await asyncio.to_thread(
            self.run,
            paramter=param
        )