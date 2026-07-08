import torch
from transformers import TrainingArguments
from trl import SFTConfig

def build_training_args(config: dict, dataset_type: str = "big", do_eval: bool = True, train_type: str = "sft") -> SFTConfig:
    """Сборка SFTConfig из YAML-конфига.

    dataset_type: "micro" -- микро-датасет (few-shot, epoch-чеки),
                  "big" -- большой датасет (steps-чеки).
    train_type: "sft" -- instruction fine-tuning,
                "cpt" -- continual pre-training.
    do_eval: False -- отключить валидацию (например, при обучении по одному файлу).
    """
    training = config.get("training", {})
    dataset = config.get("dataset", {})

    if dataset_type == "micro":
        save_strategy = "epoch"
        save_steps = None
        logging_steps = 2
        eval_steps = 2
    else:
        save_strategy = "steps"
        save_steps = 100
        logging_steps = 10
        eval_steps = 50

    # CPT использует другие гиперпараметры
    if train_type == "cpt":
        cpt_cfg = config.get("cpt", {})
        learning_rate = cpt_cfg.get("learning_rate", 1.0e-4)
        lr_scheduler_type = cpt_cfg.get("lr_scheduler_type", "cosine")
        warmup_ratio = cpt_cfg.get("warmup_ratio", 0.05)
    else:
        learning_rate = training.get("learning_rate", 2e-4)
        lr_scheduler_type = training.get("lr_scheduler_type", "constant")
        warmup_ratio = training.get("warmup_ratio", 0.03)

    args = {
        "output_dir": training.get("output_dir", "./output"),
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
        "learning_rate": learning_rate,
        "logging_steps": logging_steps,
        "num_train_epochs": training.get("num_train_epochs", 1),
        "optim": training.get("optim", "paged_adamw_32bit"),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "fp16": training.get("fp16", False),
        "bf16": training.get("bf16", True),
        "max_grad_norm": training.get("max_grad_norm", 0.3),
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "report_to": training.get("report_to", "none"),
        "max_seq_length": training.get("max_seq_length", 2048),
        "dataset_text_field": training.get("dataset_text_field", "text"),
        # Evaluation
        "do_eval": do_eval,
        "eval_strategy": training.get("eval_strategy", "steps") if do_eval else "no",
        "eval_steps": eval_steps,
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
    }

    return SFTConfig(**args)
