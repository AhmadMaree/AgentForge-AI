"""Lightweight Agentic RL dataset and reward utilities.

Training backends (Transformers/TRL/PEFT) are intentionally optional.
"""
from agentforge.rl.datasets import (
    GSM8KDataset,
    create_rl_dataset,
    create_sft_dataset,
    load_gsm8k_jsonl,
)
from agentforge.rl.training import SFTTrainerWrapper, SFTTrainingConfig
from agentforge.rl.reward import (
    AccuracyReward,
    LengthPenaltyReward,
    MathRewardFunction,
    StepReward,
)

from agentforge.rl.grpo import (
    GRPOTrainerWrapper,
    GRPOTrainingConfig,
    BatchRewardAdapter
)

__all__ = [
    "GSM8KDataset",
    "create_rl_dataset",
    "create_sft_dataset",
    "load_gsm8k_jsonl",
    "AccuracyReward",
    "LengthPenaltyReward",
    "MathRewardFunction",
    "StepReward",
    "SFTTrainingConfig",
    "SFTTrainerWrapper",
    "BatchRewardAdapter",
    "GRPOTrainingConfig",
    "GRPOTrainerWrapper",
]
