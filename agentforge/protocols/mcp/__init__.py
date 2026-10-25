"""MCP client integration for AgentForge."""
from .client import (
    MCPAdapterError,
    MCPClientAdapter,
    MCPConnectionError,
    MCPConfigurations,
    MCPToolDefinition,
    MCPToolError,
)
from .tools import (
    MCPBackgroundRunner, MCPToolProxy, close_mcp_tools_sync,
    connect_mcp_tools_sync, discover_mcp_tools, register_mcp_tools,
)

__all__ = [
    "MCPAdapterError",
    "MCPClientAdapter",
    "MCPConnectionError",
    "MCPConfigurations",
    "MCPToolDefinition",
    "MCPToolError",
    "MCPBackgroundRunner",
    "MCPToolProxy",
    "close_mcp_tools_sync",
    "connect_mcp_tools_sync",
    "discover_mcp_tools",
    "register_mcp_tools",
]
