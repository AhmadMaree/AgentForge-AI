

import logging
import asyncio


from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from contextlib import AsyncExitStack

loggers = logging.Logger(__name__)

class MCPAdapterError(RuntimeError):
    """Base error for MCP adapter failures."""


class MCPConnectionError(MCPAdapterError):
    """Raised when connecting to an MCP server fails."""


class MCPToolError(MCPAdapterError):
    """Raised when MCP tool discovery or execution fails."""


@dataclass(frozen=True, slots=True)
class MCPConfigurations:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MCP server name cannot be empty")
        if not self.command.strip():
            raise ValueError("MCP server command cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
    

class MCPToolDefinition:

    name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any]= field(default_factory=dict)
    server_name: str  = ""


class MCPClientAdapter:

    def __init__(self, config: MCPConfigurations):
        self.config = config
        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._lock = asyncio.Lock()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._session is not None

    async def __aenter__(self) -> "MCPClientAdapter":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connected:
            return

        stack = AsyncExitStack()
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                cwd=self.config.cwd,
                env= self.config.env if self.config.env is not None else None
            )

            read_stream,write_stream = await stack.enter_async_context(
                stdio_client(params)
            )

            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await asyncio.wait_for(
                session.initialize(),
                timeout = self.config.timeout_seconds
            )

        except Exception as exc:
            raise MCPConnectionError(f"Could not connect to MCP server '{self.config.name}'")


        self._stack = stack
        self._session = session
        self._connected = True
        loggers.info("Connected to MCP server '%s'",self.config.name)


    async def close(self) -> None:
        """Close the session and all transport resources safely."""
        stack, self._stack = self._stack, None
        self._session = None
        self._connected = False
        if stack is not None:
            await stack.aclose()
        loggers.info("Closed MCP server '%s'", self.config.name)


    def _require_session(self) -> Any:
        if not self.connected:
            raise MCPConnectionError(
                f"MCP server '{self.config.name}' is not connected; "
                "use `async with client` or call `await client.connect()` first"
            )
        return self._session

    async def list_tools(self) -> list[MCPToolDefinition]:
        session = self._require_session()
        try:
            async with self._lock:
                response = await asyncio.wait_for(
                    session.list_tools(), timeout=self.config.timeout_seconds
                )
            result: list[MCPToolDefinition] = []

            for item in getattr(response, "tools", []) or []:
                raw_name = str(getattr(item, "name", "")).strip()
                if not raw_name:
                    loggers.warning("Ignoring MCP tool with an empty name")
                    continue
                result.append(
                    MCPToolDefinition(
                        name=raw_name,
                        qualified_name=f"{self.config.name}_{raw_name}",
                        description=str(getattr(item, "description", "") or ""),
                        input_schema=dict(
                            getattr(item, "inputSchema", None)
                            or getattr(item, "input_schema", None)
                            or {"type": "object", "properties": {}}
                        ),
                        server_name=self.config.name,
                    )
                )
            return result
        except Exception as exc:
            raise MCPToolError(
                f"Tool discovery failed for MCP server '{self.config.name}': {exc}"
            ) from exc
        pass
   
   
    async def call_tool(
        self, tool_name: str, arguments: Mapping[str, Any] | None = None
    ) -> str:
        """Execute a server tool and return a stable text representation.

        MCP tool results commonly contain `content` blocks. Text blocks are
        joined; structured content is JSON-serialized when no text is present.
        MCP `isError` results are raised rather than silently treated as success.
        """
        session = self._require_session()
        name = tool_name.strip()
        if not name:
            raise ValueError("tool_name cannot be empty")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping/dictionary")

        try:
            async with self._lock:
                response = await asyncio.wait_for(
                    session.call_tool(name, arguments=dict(arguments or {})),
                    timeout=self.config.timeout_seconds,
                )
        except Exception as exc:
            raise MCPToolError(
                f"MCP tool '{self.config.name}_{name}' failed: {exc}"
            ) from exc

        if getattr(response, "isError", False):
            detail = self._content_to_text(getattr(response, "content", []))
            raise MCPToolError(
                f"MCP tool '{self.config.name}_{name}' returned an error: {detail}"
            )
        text = self._content_to_text(getattr(response, "content", []))
        if text:
            return text

        structured = getattr(response, "structuredContent", None)
        if structured is None:
            structured = getattr(response, "structured_content", None)
        if structured is not None:
            import json
            return json.dumps(structured, ensure_ascii=False, default=str)
        return ""
   
    @staticmethod
    def _content_to_text(content: Sequence[Any] | None) -> str:
        pieces: list[str] = []
        for block in content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                pieces.append(text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                pieces.append(block["text"])
        return "\n".join(pieces).strip()


