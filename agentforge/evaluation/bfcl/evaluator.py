
"""Evaluate BFCL-style function-calling cases."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .matcher import calls_match, parse_calls


def evaluate(
    cases: list[dict],
    predict: Callable[[dict], Any],
    *,
    ordered: bool = False,
) -> dict:
    results = []

    for case in cases:
        try:
            prediction = parse_calls(predict(case))

            correct = calls_match(
                prediction,
                case["ground_truth"],
                ordered=ordered,
            )

            error = None

        except Exception as exc:
            prediction = []
            correct = False
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        results.append(
            {
                "id": case["id"],
                "category": case.get(
                    "category",
                    case["id"].split("_")[0],
                ),
                "correct": correct,
                "prediction": prediction,
                "ground_truth": case["ground_truth"],
                "error": error,
            }
        )

    by_category = defaultdict(list)

    for result in results:
        by_category[result["category"]].append(
            result
        )

    total = len(results)
    correct_count = sum(
        result["correct"] for result in results
    )

    return {
        "total_samples": total,
        "correct_samples": correct_count,
        "overall_accuracy": (
            correct_count / total if total else 0.0
        ),
        "category_accuracy": {
            category: (
                sum(item["correct"] for item in items)
                / len(items)
            )
            for category, items in sorted(
                by_category.items()
            )
        },
        "results": results,
    }


def save_report(
    report: dict,
    destination: str | Path,
) -> None:
    path = Path(destination)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# AgentForge BFCL-style Evaluation",
        "",
        (
            f"Accuracy: "
            f"{report['overall_accuracy']:.2%} "
            f"({report['correct_samples']}/"
            f"{report['total_samples']})"
        ),
        "",
        "## Categories",
    ]

    lines.extend(
        f"- {name}: {accuracy:.2%}"
        for name, accuracy
        in report["category_accuracy"].items()
    )

    lines += ["", "## Failures"]

    lines.extend(
        (
            f"- {result['id']}: "
            f"{result['error'] or 'Incorrect tool calls'}"
        )
        for result in report["results"]
        if not result["correct"]
    )

    return "\n".join(lines) + "\n"
