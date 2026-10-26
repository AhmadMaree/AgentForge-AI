from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any



_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_FINAL_RE = re.compile(r"(?:####|final answer\s*:\s*)\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)

def _normalize_answer(value: Any) -> str:
    text = str(value).strip().replace(",", "")
    # Normalize a simple numeric answer, preserving non-numeric symbolic answers.
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return re.sub(r"\s+", " ", text).casefold().rstrip(".")
    if number == number.to_integral_value():
        return str(number.quantize(Decimal("1")))
    return format(number.normalize(), "f")


def extract_final_answer(response: str) -> str:
    """Extract a final answer marker, falling back to the last numeric token."""
    if not isinstance(response, str):
        return ""
    text = response.strip()
    marked = _FINAL_RE.search(text)
    if marked:
        candidate = marked.group(1).strip().splitlines()[0].strip()
        # Avoid capturing trailing prose after a short numeric answer.
        numeric = _NUMBER_RE.search(candidate)
        return numeric.group(0) if numeric else candidate.rstrip(".")
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1] if numbers else text.splitlines()[-1].strip() if text else ""

@dataclass(frozen=True)
class AccuracyReward:
    """Returns 1.0 for a matching final answer and 0.0 otherwise."""

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0

    def __call__(self, response: str, ground_truth: str, **_: Any) -> float:
        predicted = _normalize_answer(extract_final_answer(response))
        expected = _normalize_answer(extract_final_answer(ground_truth))
        return self.correct_reward if predicted == expected else self.incorrect_reward


@dataclass(frozen=True)
class LengthPenaltyReward:
    """Penalize responses longer than a configured character budget."""

    max_length: int = 1200
    penalty_per_100_chars: float = 0.05
    minimum_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.max_length < 0:
            raise ValueError("max_length must be non-negative")
        if self.penalty_per_100_chars < 0:
            raise ValueError("penalty_per_100_chars must be non-negative")

    def __call__(self, response: str, **_: Any) -> float:
        excess = max(0, len(response) - self.max_length)
        penalty = (excess / 100.0) * self.penalty_per_100_chars
        return max(self.minimum_reward, -penalty)



@dataclass(frozen=True)
class StepReward:
    """Reward responses that contain explicit reasoning steps and a final answer.

    This is a lightweight format heuristic, not a semantic proof of reasoning.
    """

    step_reward: float = 0.1
    final_answer_reward: float = 0.2
    max_step_rewards: int = 5

    def __post_init__(self) -> None:
        if self.max_step_rewards < 0:
            raise ValueError("max_step_rewards must be non-negative")

    def __call__(self, response: str, ground_truth: str | None = None, **_: Any) -> float:
        text = response or ""
        numbered_steps = len(re.findall(r"(?m)^\s*(?:step\s*)?\d+[.)]\s+", text, re.IGNORECASE))
        has_reasoning = len(text.split()) >= 8
        steps = min(numbered_steps, self.max_step_rewards)
        reward = steps * self.step_reward
        if has_reasoning and steps == 0:
            reward += self.step_reward
        if _FINAL_RE.search(text) or re.search(r"(?i)\banswer\s+is\b", text):
            reward += self.final_answer_reward
        return reward



@dataclass(frozen=True)
class MathRewardFunction:
    """Combine accuracy, response-format/step reward, and length penalty."""

    accuracy: AccuracyReward = AccuracyReward()
    step: StepReward = StepReward()
    length_penalty: LengthPenaltyReward = LengthPenaltyReward()

    def __call__(
        self,
        response: str,
        ground_truth: str,
        **kwargs: Any,
    ) -> float:
        return (
            self.accuracy(response, ground_truth, **kwargs)
            + self.step(response, ground_truth=ground_truth, **kwargs)
            + self.length_penalty(response, ground_truth=ground_truth, **kwargs)
        )