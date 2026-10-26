"""Optional SFT training support for AgentForge.

Heavy training dependencies are imported lazily so dataset/reward utilities remain
usable on CPU-only machines without installing the ML stack.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SFTTrainingConfig:
    """Configuration for supervised fine-tuning of a causal language model."""

    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./models/sft_model"
    num_epochs: float = 3.0
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_samples: int | None = 100
    max_seq_length: int = 1024
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    save_strategy: str = "epoch"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    report_to: str = "none"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be greater than zero")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be None or at least 1")
        if self.max_seq_length < 8:
            raise ValueError("max_seq_length must be at least 8")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("lora_rank and lora_alpha must be at least 1")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")


class SFTTrainerWrapper:
    """Train a causal LM on ``prompt``/``completion`` examples using TRL.

    ``model`` and ``tokenizer`` can be supplied for tests or custom loading. In
    ordinary use they are loaded from ``config.model_name``. TRL/Transformers/
    Datasets are optional dependencies and only required when ``train`` is called.
    """

    def __init__(self, config: SFTTrainingConfig | None = None) -> None:
        self.config = config or SFTTrainingConfig()

    @staticmethod
    def _prepare_examples(
        examples: Iterable[Mapping[str, Any]], max_samples: int | None = None
    ) -> list[dict[str, str]]:
        prepared: list[dict[str, str]] = []
        for index, example in enumerate(examples):
            if max_samples is not None and len(prepared) >= max_samples:
                break
            prompt = example.get("prompt")
            completion = example.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Example {index} has an empty or invalid prompt")
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError(f"Example {index} has an empty or invalid completion")
            prepared.append({"text": f"{prompt.rstrip()}\n{completion.lstrip()}"})
        if not prepared:
            raise ValueError("SFT training requires at least one valid example")
        return prepared

    def train(
        self,
        examples: Iterable[Mapping[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> dict[str, Any]:
        """Run SFT and return a compact summary of the saved training result."""
        try:
            from datasets import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from trl import SFTTrainer
        except ImportError as exc:
            raise ImportError(
                "SFT training requires the optional 'rl' dependencies. Install them with: "
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
                raise ImportError(
                    "LoRA was enabled but PEFT is unavailable. Install with: pip install peft"
                ) from exc
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora_config)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            report_to=cfg.report_to,
            remove_unused_columns=False,
        )
        train_dataset = Dataset.from_list(prepared)

        # TRL has changed constructor names between releases. Only pass kwargs
        # supported by the installed version to provide a useful compatibility path.
        signature = inspect.signature(SFTTrainer.__init__)
        supported = signature.parameters
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
        }
        if "processing_class" in supported:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in supported:
            trainer_kwargs["tokenizer"] = tokenizer
        if "dataset_text_field" in supported:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in supported:
            trainer_kwargs["max_seq_length"] = cfg.max_seq_length
        elif "max_length" in supported:
            trainer_kwargs["max_length"] = cfg.max_seq_length

        trainer = SFTTrainer(**trainer_kwargs)
        result = trainer.train()
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        metrics = getattr(result, "metrics", {}) or {}
        loss = metrics.get("train_loss")
        return {
            "model_path": cfg.output_dir,
            "num_samples": len(prepared),
            "num_epochs": cfg.num_epochs,
            "final_loss": float(loss) if loss is not None else None,
            "metrics": dict(metrics),
        }
