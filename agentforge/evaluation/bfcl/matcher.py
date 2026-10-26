
"""Safe structural matching for function calls."""

import ast
import json
from collections import Counter
from typing import Any


def _value(node: ast.AST) -> Any:
    """Read safe literals and basic numeric expressions."""

    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div),
    ):
        left = _value(node.left)
        right = _value(node.right)

        if type(left) not in (int, float) or type(right) not in (
            int,
            float,
        ):
            raise ValueError(
                "Only numeric arithmetic is supported"
            )

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        if isinstance(node.op, ast.Mult):
            return left * right

        if right == 0:
            raise ValueError("Division by zero")

        return left / right

    try:
        return ast.literal_eval(node)
    except (
        ValueError,
        TypeError,
        SyntaxError,
        RecursionError,
    ) as exc:
        raise ValueError(
            "Unsafe or unsupported argument expression"
        ) from exc


def parse_calls(prediction: Any) -> list[dict]:
    """Parse JSON or restricted Python function calls."""

    if isinstance(prediction, str):
        text = prediction.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if (
                len(lines) < 3
                or not lines[-1].strip().startswith("```")
            ):
                raise ValueError("Malformed code fence")

            text = "\n".join(lines[1:-1]).strip()

        try:
            prediction = json.loads(text)

        except json.JSONDecodeError:
            tree = ast.parse(text, mode="exec")
            calls = []

            for statement in tree.body:
                if (
                    not isinstance(statement, ast.Expr)
                    or not isinstance(
                        statement.value,
                        ast.Call,
                    )
                ):
                    raise ValueError(
                        "Expected only function call expressions"
                    )

                call = statement.value

                if (
                    not isinstance(call.func, ast.Name)
                    or call.args
                ):
                    raise ValueError(
                        "Expected named functions with "
                        "keyword arguments"
                    )

                arguments = {}

                for keyword in call.keywords:
                    if (
                        keyword.arg is None
                        or keyword.arg in arguments
                    ):
                        raise ValueError(
                            "Invalid keyword argument"
                        )

                    arguments[keyword.arg] = _value(
                        keyword.value
                    )

                calls.append(
                    {
                        "name": call.func.id,
                        "arguments": arguments,
                    }
                )

            prediction = calls

    if isinstance(prediction, dict):
        prediction = [prediction]

    if not isinstance(prediction, list):
        raise ValueError("Expected a list of calls")

    for call in prediction:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not isinstance(
                call.get("arguments"),
                dict,
            )
        ):
            raise ValueError(
                "Each call needs name and arguments object"
            )

    return prediction


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def calls_match(
    prediction: Any,
    expected: Any,
    *,
    ordered: bool = False,
) -> bool:
    """Compare function calls structurally."""

    predicted = parse_calls(prediction)
    actual = parse_calls(expected)

    predicted_values = [
        _canonical(call) for call in predicted
    ]

    actual_values = [
        _canonical(call) for call in actual
    ]

    if ordered:
        return predicted_values == actual_values

    return (
        Counter(predicted_values)
        == Counter(actual_values)
    )
