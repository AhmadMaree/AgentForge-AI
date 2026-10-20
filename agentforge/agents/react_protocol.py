
import json

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


class ReActProtocolError(ValueError):
    """The model returned an invalid ReAct action."""


class ReActDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool", "final"]

    tool_name: str | None = None

    arguments: dict[str, Any] | None = None

    answer: str | None = None

    @model_validator(mode="after")
    def validate_decision(self):
        if self.type == "tool":
            if not self.tool_name:
                raise ValueError(
                    "A tool decision requires tool_name."
                )

            if self.arguments is None:
                raise ValueError(
                    "A tool decision requires arguments."
                )

            if self.answer is not None:
                raise ValueError(
                    "A tool decision cannot include answer."
                )

        if self.type == "final":
            if not self.answer or not self.answer.strip():
                raise ValueError(
                    "A final decision requires an answer."
                )

            if (
                self.tool_name is not None
                or self.arguments is not None
            ):
                raise ValueError(
                    "A final decision cannot request a tool."
                )

        return self


def parse_react_decision(
    raw_text: str,
) -> ReActDecision:
    try:
        payload = json.loads(raw_text)

        return ReActDecision.model_validate(
            payload
        )

    except (json.JSONDecodeError, ValidationError) as exc:
        raise ReActProtocolError(
            f"Invalid ReAct response: {exc}"
        ) from exc
