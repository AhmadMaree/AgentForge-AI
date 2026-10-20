
import ast
import math
import operator

from typing import ClassVar

from pydantic import BaseModel, Field, ConfigDict

from agentforge.tools.base import Tool


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Arithmetic expression, e.g. 15 * 8 + 32"
        ),
    )


class CalculatorTool(Tool):
    name: ClassVar[str] = "calculator"

    description: ClassVar[str] = (
        "Evaluate arithmetic expressions using "
        "+, -, *, /, //, %, ** and parentheses."
    )

    input_schema: ClassVar[type[BaseModel]] = CalculatorInput

    OPERATORS: ClassVar[dict] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    MAX_ABS_VALUE: ClassVar[float] = 1e12
    MAX_POWER: ClassVar[int] = 10
    MAX_NODES: ClassVar[int] = 40

    def _excute(
        self,
        parameters: CalculatorInput,
    ) -> float | int:
        try:
            tree = ast.parse(
                parameters.expression,
                mode="eval",
            )
        except SyntaxError as exc:
            raise ValueError(
                "Invalid arithmetic expression."
            ) from exc

        if sum(1 for _ in ast.walk(tree)) > self.MAX_NODES:
            raise ValueError("Expression is too complex.")

        return self._evaluate(tree.body)

    def _check_number(
        self,
        value: int | float,
    ) -> int | float:
        if not isinstance(value, (int, float)):
            raise ValueError("Unsupported value.")

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite result.")

        if abs(value) > self.MAX_ABS_VALUE:
            raise ValueError("Number exceeds safe limit.")

        return value

    def _evaluate(
        self,
        node: ast.AST,
    ) -> int | float:
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numbers are allowed.")

            return self._check_number(node.value)

        if isinstance(node, ast.UnaryOp):
            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported unary operator."
                )

            operand = self._evaluate(node.operand)
            return self._check_number(
                operation(operand)
            )

        if isinstance(node, ast.BinOp):
            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported binary operator."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > self.MAX_POWER:
                    raise ValueError(
                        "Exponent exceeds safe limit."
                    )

                # Bound the base before exponentiation.
                if abs(left) > 1 and (
                    abs(left) ** abs(right)
                    > self.MAX_ABS_VALUE
                ):
                    raise ValueError(
                        "Power result exceeds safe limit."
                    )

            return self._check_number(
                operation(left, right)
            )

        raise ValueError(
            f"Unsupported expression: "
            f"{type(node).__name__}"
        )
