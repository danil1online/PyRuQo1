def format_single_example(example: dict) -> dict:
    """Форматирование одного примера датасета в единое поле 'text'."""
    return {
        "text": (
            f"Система: {example['system']}\n\n"
            f"Пользователь: {example['prompt']}\n\n"
            f"Система: {example['response']}"
        )
    }


def formatting_prompts_func(examples: dict) -> list:
    """Функция форматирования для SFTTrainer (batch mode)."""
    output_texts = []
    for i in range(len(examples.get('prompt', []))):
        text = (
            f"Система: {examples['system'][i]}\n\n"
            f"Пользователь: {examples['prompt'][i]}\n\n"
            f"Система: {examples['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def format_dataset(dataset, remove_columns: list = None) -> dict:
    """
    Применить форматирование к датасету (map + remove_columns).
    Возвращает отформатированный датасет с полем 'text'.
    """
    if remove_columns is None:
        remove_columns = list(dataset.column_names)

    formatted = dataset.map(
        format_single_example,
        remove_columns=remove_columns,
        desc="Форматирование датасета",
    )
    return formatted
