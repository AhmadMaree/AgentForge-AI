
"""Load local BFCL-style JSON and JSONL datasets."""

import json
from pathlib import Path


def load_cases(path: str | Path) -> list[dict]:
    path = Path(path)

    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(
            path.read_text(encoding="utf-8")
        )

    if not isinstance(records, list):
        raise ValueError(
            "Expected a JSON array or JSONL records"
        )

    seen = set()

    for index, item in enumerate(records):
        required = (
            "id",
            "question",
            "function",
            "ground_truth",
        )

        if (
            not isinstance(item, dict)
            or not all(key in item for key in required)
        ):
            raise ValueError(
                f"Invalid case at index {index}"
            )

        case_id = item["id"]

        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
        ):
            raise ValueError(
                f"Duplicate or invalid ID at index {index}"
            )

        seen.add(case_id)

        if (
            not isinstance(item["question"], str)
            or not isinstance(item["function"], list)
            or not isinstance(item["ground_truth"], list)
        ):
            raise ValueError(
                f"Invalid fields for case {case_id}"
            )

    return records
