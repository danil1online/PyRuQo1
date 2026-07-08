def format_single_example_default(example: dict) -> dict:
    return {
        "text": (
            f"Система: {example['system']}\n\n"
            f"Пользователь: {example['prompt']}\n\n"
            f"Система: {example['response']}"
        )
    }


def format_single_example_chatml(example: dict) -> dict:
    return {
        "text": (
            f"<|system|>\n{example['system']}<|end|>\n"
            f"<|user|>\n{example['prompt']}<|end|>\n"
            f"<|assistant|>\n{example['response']}<|end|>"
        )
    }


def formatting_prompts_func_default(examples: dict) -> list:
    output_texts = []
    for i in range(len(examples.get('prompt', []))):
        text = (
            f"Система: {examples['system'][i]}\n\n"
            f"Пользователь: {examples['prompt'][i]}\n\n"
            f"Система: {examples['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def formatting_prompts_func_chatml(examples: dict) -> list:
    output_texts = []
    for i in range(len(examples.get('prompt', []))):
        text = (
            f"<|system|>\n{examples['system'][i]}<|end|>\n"
            f"<|user|>\n{examples['prompt'][i]}<|end|>\n"
            f"<|assistant|>\n{examples['response'][i]}<|end|>"
        )
        output_texts.append(text)
    return output_texts


def format_single_example_cpt(example: dict) -> dict:
    return {"text": example["text"]}


def formatting_prompts_func_cpt(examples: dict) -> list:
    text = examples.get("text", [])
    if isinstance(text, list):
        return text
    return [text]


def format_dataset(dataset, remove_columns: list = None, format_type: str = "default") -> dict:
    if format_type == "chatml":
        format_fn = format_single_example_chatml
    elif format_type == "cpt":
        format_fn = format_single_example_cpt
    else:
        format_fn = format_single_example_default

    if remove_columns is None:
        remove_columns = list(dataset.column_names)

    formatted = dataset.map(
        format_fn,
        remove_columns=remove_columns,
        desc="Форматирование датасета",
    )
    return formatted


# Backwards compatibility aliases
format_single_example = format_single_example_default
formatting_prompts_func = formatting_prompts_func_default
