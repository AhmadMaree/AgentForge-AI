# AgentForge AI

**A modular AI agent frameworks build with Python from scratch.**

AgentForge is a custom frameworks for understaning and building AI agent without relying on high-level agent orchestration libraries.

## Goals
 - Build a unified interface for multiple LLM providers.
 - Integrate Google Gemini and local models
 - Implement Simple, ReAct, Reflection, and Plan-and-Solve agents
 - Create a reusable tool registry
 - Support calculator and web search tools (Tavily web search)
 - Add structured function calling, streaming, and asynchronous execution
 - Develop a fully tested and documented Python package

## Tech Stack

Python 3.11+, Google Gemini, Pydantic, Pytest, Tavily, and Ollama.

## Getting Started

```bash
python -m venv .venv
pip install -e ".[dev,search]"
python -m pytest -v
```

Copy `.env.example` to `.env` and configure your API keys.


## License

License to be selected before public distribution.