
"""Run BFCL-style predictions using AgentForge agents."""

import json
from typing import Any

from .matcher import parse_calls


def predict_with_agent(
    agent: Any,
    case: dict,
) -> list[dict]:
    functions = json.dumps(
        case["function"],
        ensure_ascii=False,
    )

    prompt = (
        "Select function calls for the user request. "
        "Do not execute functions. "
        "Return ONLY a JSON array of objects with "
        "'name' and 'arguments'. "
        "Return [] if no function is relevant.\n"
        f"Available functions: {functions}\n"
        f"User request: {case['question']}"
    )

    # Prevent history from previous cases
    # affecting the current evaluation.
    if hasattr(agent, "reset"):
        agent.reset()

    response = agent.run(prompt)

    return parse_calls(response)
