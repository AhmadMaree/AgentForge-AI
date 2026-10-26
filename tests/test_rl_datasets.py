import pytest

from agentforge.rl.datasets import GSM8KDataset, create_rl_dataset


def test_from_records_normalizes_all_records():
    records = [
        {"question": " What is 2 + 2? ", "answer": " 4\n#### 4 "},
        {"question": "What is 3 + 5?", "answer": "8\n#### 8"},
    ]

    dataset = GSM8KDataset.from_records(records)

    assert dataset.examples == (
        {"question": "What is 2 + 2?", "answer": "4\n#### 4"},
        {"question": "What is 3 + 5?", "answer": "8\n#### 8"},
    )
    assert len(create_rl_dataset(records)) == 2


def test_from_records_rejects_non_mapping_record():
    with pytest.raises(TypeError, match="Record 0 must be a mapping"):
        GSM8KDataset.from_records(["not a mapping"])
