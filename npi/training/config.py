import torch
from transformers import TrainingArguments, SFTConfig


def build_training_args(config: dict) -> SFTConfig:
    """Сборка SFTConfig из YAML-конфига."""
    training = config.get("training", {})
    dataset = config.get("dataset", {})

    args = {
        "output_dir": training.get("output_dir", "./output"),
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
        "learning_rate": training.get("learning_rate", 2e-4),
        "logging_steps": training.get("logging_steps", 10),
        "num_train_epochs": training.get("num_train_epochs", 1),
        "optim": training.get("optim", "paged_adamw_32bit"),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "fp16": training.get("fp16", False),
        "bf16": training.get("bf16", True),
        "max_grad_norm": training.get("max_grad_norm", 0.3),
        "warmup_ratio": training.get("warmup_ratio", 0.03),
        "lr_scheduler_type": training.get("lr_scheduler_type", "constant"),
        "save_strategy": training.get("save_strategy", "steps"),
        "save_steps": training.get("save_steps", 100),
        "report_to": training.get("report_to", "none"),
        "max_seq_length": training.get("max_seq_length", 2048),
        "dataset_text_field": training.get("dataset_text_field", "text"),
        # Evaluation
        "do_eval": training.get("do_eval", True),
        "eval_strategy": training.get("eval_strategy", "steps"),
        "eval_steps": training.get("eval_steps", 50),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
    }

    return SFTConfig(**args)
