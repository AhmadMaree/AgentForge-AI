import re

from typing import Any

def toknizer(text: str) -> set[str]:
    return set(
        re.findall(r"\w+", text.casefold())
    )



def _call_tool(tool: Any, parameters: dict[str, Any]) -> Any:
        """Call a supported tool entry point without assuming its backend."""

        run = getattr(tool, "run", None)
        if callable(run):
            return run(parameters)

        execute = getattr(tool, "execute", None)
        if callable(execute):
            # Some AgentForge tools use execute(action, **kwargs);
            # others use execute(dict). Try the dictionary form first.
            try:
                return execute(parameters)
            except TypeError:
                action = parameters.get("action")
                remaining = {
                    key: value
                    for key, value in parameters.items()
                    if key != "action"
                }
                if action is not None:
                    return execute(action, **remaining)
                raise

        raise TypeError(
            f"{type(tool).__name__} exposes neither run() nor execute()"
        )
