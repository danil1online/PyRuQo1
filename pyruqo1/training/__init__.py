def __getattr__(name):
    if name == "NPITrainer":
        from pyruqo1.training.trainer import NPITrainer
        return NPITrainer
    elif name == "format_single_example":
        from pyruqo1.training.formatting import format_single_example
        return format_single_example
    elif name == "formatting_prompts_func":
        from pyruqo1.training.formatting import formatting_prompts_func
        return formatting_prompts_func
    elif name == "format_dataset":
        from pyruqo1.training.formatting import format_dataset
        return format_dataset
    elif name == "build_training_args":
        from pyruqo1.training.config import build_training_args
        return build_training_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NPITrainer",
    "format_single_example",
    "formatting_prompts_func",
    "format_dataset",
    "build_training_args",
]
