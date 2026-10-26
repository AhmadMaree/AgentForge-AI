from agentforge.rl import (
    GRPOTrainerWrapper,
    GRPOTrainingConfig,
    create_rl_dataset,
)
from typing import Mapping, Iterable

records: Iterable[Mapping[str,str]] = [
    {
        "question": "What is 2 + 2?",
        "answer": "2 + 2 = 4.\n#### 4",
    },
    {
        "question": "What is 3 + 5?",
        "answer": "3 + 5 = 8.\n#### 8",
    },
]

dataset = create_rl_dataset(records)

config = GRPOTrainingConfig(
    model_name="Qwen/Qwen3-0.6B",
    output_dir="./models/grpo_model",
    batch_size=2,
    max_samples=2,
    num_generations=2,
    use_lora=True,
    use_cpu=False,
)

trainer = GRPOTrainerWrapper(config)
result = trainer.train(dataset)

print(result)