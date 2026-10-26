from dataclasses import dataclass
from typing import Iterable, Any, Mapping

import re
import json
from pathlib import Path


_FINAL_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)

@dataclass(frozen=True)
class GSM8KDataset:

    examples: tuple[dict[str, str], ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str,Any]]
    ) -> "GSM8KDataset":
        normalized: list[dict[str, str]] = []

        for index, rec in enumerate(records):
            if not isinstance(rec, Mapping):
                raise TypeError(f"Record {index} must be a mapping")

            question = rec.get("question")
            answer = rec.get("answer")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Record {index} has an empty or invalid question")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Record {index} has an empty or invalid answer")

            normalized.append({
                "question": question.strip(),
                "answer": answer.strip(),
            })

        return cls(tuple(normalized))
    
    def to_sft(self) -> list[dict[str, str]]:
        return create_sft_dataset(self.examples)

    def to_rl(self) -> list[dict[str, str]]:
        return create_rl_dataset(self.examples)



def create_sft_dataset(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:

    datasets = GSM8KDataset.from_records(records)

    return [
        {
            "prompt": (
                "Solve the following math problem step by step. "
                "End with the final answer.\n\n"
                f"Problem: {item['question']}\n\nSolution:"
            ),
            "completion": item["answer"],
        }
        for item in datasets.examples
    ]



def create_rl_dataset(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Convert records to prompt/ground_truth pairs for reward-based training."""

    dataset = GSM8KDataset.from_records(records)
    return [
        {
            "prompt":(
                "Solve the following math problem. Show your reasoning and "
                "finish with the final answer.\n\n"
                f"Problem: {item['question']}\n\nAnswer:"
            ),
            "ground_truth": extract_ground_truth(item["answer"])
        }
        for item in dataset.examples
    ]



def extract_ground_truth(answer: str) -> str:
    """Extract the final answer from GSM8K's ``#### answer`` convention.

    If the marker is absent, the trimmed input is returned so the helper also
    works with already-normalized records.
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    match = _FINAL_ANSWER_RE.search(answer.strip())
    return match.group(1).strip() if match else answer.strip()


def load_gsm8k_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Load GSM8K-shaped records from a local JSON Lines file."""
    file_path = Path(path)
    records: list[dict[str, str]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {file_path}"
                ) from exc
            records.append(record)
    return list(GSM8KDataset.from_records(records).examples)
        