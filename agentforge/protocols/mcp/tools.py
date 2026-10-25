"""Thin tool proxies that expose MCP tools to AgentForge-style registries.

The proxy deliberately uses the common `name`, `description`, `run(dict)`
contract. Exact Tool base-class/schema conversion belongs in the small registry
bridge for the current repository, since AgentForge's edited base class must
not be replaced by a tutorial-derived version.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Coroutine

from .client import MCPClientAdapter, MCPToolDefinition


class MCPBackgroundRunner:
    """Own a persistent event loop for synchronous AgentForge agents.

    MCP sessions are loop-bound, so all connect/list/call/close operations for
    a given client must execute on this same background loop.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait()
        self._closed = False

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._closed:
            coroutine.close()
            raise RuntimeError("MCP background runner is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


class MCPToolProxy:
    """Expose one discovered MCP tool as a synchronous AgentForge tool.

    ReActAgent and the current registry can call `run(parameters)` as they do
    for local tools. A dedicated background event loop keeps the MCP session
    alive across individual synchronous `run` calls.
    """

    def __init__(self, client: MCPClientAdapter, definition: MCPToolDefinition, runner: MCPBackgroundRunner | None = None) -> None:
        self.client = client
        self.definition = definition
        self.name = definition.qualified_name
        self.description = definition.description or f"MCP tool {definition.name}"
        self.input_schema = definition.input_schema
        self.runner = runner

    def get_parameters(self) -> list[dict[str, Any]]:
        """Return basic parameter metadata for introspection/adaptation."""
        schema = self.input_schema or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        return [
            {
                "name": name,
                "type": spec.get("type", "any"),
                "description": spec.get("description", ""),
                "required": name in required,
                "default": spec.get("default"),
            }
            for name, spec in properties.items()
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        if not isinstance(parameters, dict):
            raise TypeError("MCP tool parameters must be a dictionary")
        coroutine = self.client.call_tool(self.definition.name, parameters)
        if self.runner is not None:
            return self.runner.run(coroutine)
        # Without a persistent runner, only use this for clients whose session
        # was created on the current event loop. For sync ReActAgent use the
        # `connect_mcp_tools_sync` helper below.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        coroutine.close()
        raise RuntimeError(
            "MCPToolProxy.run() cannot nest an event loop; use "
            "connect_mcp_tools_sync() for synchronous agents"
        )


async def discover_mcp_tools(client: MCPClientAdapter) -> list[MCPToolProxy]:
    """Connect if needed and return proxies for every discovered tool."""
    if not client.connected:
        await client.connect()
    definitions = await client.list_tools()
    return [MCPToolProxy(client, definition) for definition in definitions]


def register_mcp_tools(registry: Any, tools: list[MCPToolProxy]) -> None:
    """Register proxies without assuming one exact registry method name.

    Supports the project-style `register(tool)` and tutorial-style
    `register_tool(tool)`. Raises loudly if neither exists.
    """
    register = getattr(registry, "register", None)
    if not callable(register):
        register = getattr(registry, "register_tool", None)
    if not callable(register):
        raise TypeError(
            f"{type(registry).__name__} has neither register() nor register_tool()"
        )
    for tool in tools:
        register(tool)


def connect_mcp_tools_sync(config: Any) -> tuple[MCPBackgroundRunner, MCPClientAdapter, list[MCPToolProxy]]:
    """Connect, discover, and return proxies for a synchronous ReActAgent.

    Keep the returned runner/client alive while the tools are registered and
    used. Call `close_mcp_tools_sync(runner, client)` when the agent is done.
    """
    from .client import MCPClientAdapter

    runner = MCPBackgroundRunner()
    client = MCPClientAdapter(config)
    try:
        runner.run(client.connect())
        definitions = runner.run(client.list_tools())
        tools = [MCPToolProxy(client, definition, runner) for definition in definitions]
        return runner, client, tools
    except Exception:
        try:
            runner.run(client.close())
        finally:
            runner.close()
        raise


def close_mcp_tools_sync(runner: MCPBackgroundRunner, client: MCPClientAdapter) -> None:
    """Close an MCP client created by `connect_mcp_tools_sync`."""
    try:
        runner.run(client.close())
    finally:
        runner.close()
