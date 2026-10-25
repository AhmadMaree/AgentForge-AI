from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentForge Calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b



@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b



@mcp.tool()
def greet(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run(transport="stdio")