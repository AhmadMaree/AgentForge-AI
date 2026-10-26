"""Optional GRPO training support built on top of TRL.

The wrapper keeps TRL imports lazy so AgentForge's dataset/reward utilities can
be used without installing a model-training stack.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agentforge.rl.reward import MathRewardFunction


@dataclass(frozen=True)
class GRPOTrainingConfig:
    """Configuration for Group Relative Policy Optimization (GRPO)."""

    model_name: str = "./models/sft_model"
    output_dir: str = "./models/grpo_model"
    learning_rate: float = 5e-6
    num_epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 512
    max_samples: int | None = 100
    beta: float = 0.04
    temperature: float = 0.8
    logging_steps: int = 5
    save_strategy: str = "epoch"
    gradient_checkpointing: bool = False
    use_lora: bool = True
    use_cpu: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    report_to: str = "none"

    def __post_init__(self) -> None:
        if not self.model_name.strip() or not self.output_dir.strip():
            raise ValueError("model_name and output_dir must not be empty")
        if self.learning_rate <= 0 or self.num_epochs <= 0:
            raise ValueError("learning_rate and num_epochs must be greater than zero")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be at least 1")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2 for group-relative training")
        if self.max_prompt_length < 1 or self.max_completion_length < 1:
            raise ValueError("prompt and completion lengths must be positive")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be None or at least 1")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("lora_rank and lora_alpha must be at least 1")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")


class BatchRewardAdapter:
    """Adapt AgentForge's scalar reward functions to TRL's batched API."""

    def __init__(self, reward_fn: Callable[..., float] | None = None) -> None:
        self.reward_fn = reward_fn or MathRewardFunction()

    @staticmethod
    def _completion_text(completion: Any) -> str:
        # TRL commonly passes strings for plain-text datasets; chat datasets may
        # supply a list of message dictionaries instead.
        if isinstance(completion, str):
            return completion
        if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
            parts: list[str] = []
            for item in completion:
                if isinstance(item, Mapping):
                    content = item.get("content", "")
                    if content:
                        parts.append(str(content))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(completion)

    def __call__(
        self,
        completions: Sequence[Any],
        ground_truth: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        if ground_truth is None:
            raise ValueError("GRPO dataset must include a 'ground_truth' column")
        if len(completions) != len(ground_truth):
            raise ValueError("completions and ground_truth must have equal lengths")
        scores: list[float] = []
        for completion, expected in zip(completions, ground_truth):
            score = self.reward_fn(
                self._completion_text(completion), str(expected), **kwargs
            )
            scores.append(float(score))
        return scores


class GRPOTrainerWrapper:
    """Train a causal language model with TRL's GRPOTrainer.

    Training dependencies are imported only when ``train`` is called. Inject
    ``model``, ``tokenizer`` and ``trainer_cls`` for controlled tests/custom use.
    """

    def __init__(
        self,
        config: GRPOTrainingConfig | None = None,
        reward_fn: Callable[..., float] | None = None,
    ) -> None:
        self.config = config or GRPOTrainingConfig()
        self.reward_adapter = BatchRewardAdapter(reward_fn)

    @staticmethod
    def _prepare_examples(
        examples: Iterable[Mapping[str, Any]], max_samples: int | None = None
    ) -> list[dict[str, str]]:
        prepared: list[dict[str, str]] = []
        for index, example in enumerate(examples):
            if max_samples is not None and len(prepared) >= max_samples:
                break
            prompt = example.get("prompt")
            truth = example.get("ground_truth")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Example {index} has an empty or invalid prompt")
            if not isinstance(truth, (str, int, float)) or not str(truth).strip():
                raise ValueError(f"Example {index} has an empty or invalid ground_truth")
            prepared.append({"prompt": prompt.strip(), "ground_truth": str(truth).strip()})
        if not prepared:
            raise ValueError("GRPO training requires at least one valid example")
        return prepared

    def train(
        self,
        examples: Iterable[Mapping[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        trainer_cls: Any | None = None,
    ) -> dict[str, Any]:
        try:
            from datasets import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise ImportError(
                "GRPO training requires the optional 'rl' dependencies. Install them with: "
                "pip install -e .[rl]"
            ) from exc

        cfg = self.config
        prepared = self._prepare_examples(examples, cfg.max_samples)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        if cfg.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError("LoRA is enabled but PEFT is unavailable; install peft") from exc
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules="all-linear",
                ),
            )

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        config_kwargs = {
            "output_dir": cfg.output_dir,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_epochs,
            "per_device_train_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_generations": cfg.num_generations,
            "max_prompt_length": cfg.max_prompt_length,
            "max_completion_length": cfg.max_completion_length,
            "beta": cfg.beta,
            "temperature": cfg.temperature,
            "logging_steps": cfg.logging_steps,
            "save_strategy": cfg.save_strategy,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "use_cpu": cfg.use_cpu,
            "report_to": cfg.report_to,
            "remove_unused_columns": False,
        }
        config_params = inspect.signature(GRPOConfig.__init__).parameters
        config_kwargs = {k: v for k, v in config_kwargs.items() if k in config_params}
        training_args = GRPOConfig(**config_kwargs)
        train_dataset = Dataset.from_list(prepared)
        cls = trainer_cls or GRPOTrainer
        trainer_params = inspect.signature(cls.__init__).parameters
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "reward_funcs": [self.reward_adapter],
        }
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = cls(**trainer_kwargs)
        result = trainer.train()
        trainer.save_model(cfg.output_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(cfg.output_dir)
        metrics = getattr(result, "metrics", {}) or {}
        loss = metrics.get("train_loss")
        return {
            "model_path": cfg.output_dir,
            "num_samples": len(prepared),
            "num_epochs": cfg.num_epochs,
            "num_generations": cfg.num_generations,
            "final_loss": float(loss) if loss is not None else None,
            "metrics": dict(metrics),
        }
