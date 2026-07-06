#!/usr/bin/env python3
"""
Базовый пример обучения на готовом датасете HuggingFace.

Сохранён для обратной совместимости.
Рекомендуемый способ: pyruqo1 train --model gigachat-20b --train-file <path>
"""

from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer

config = load_config(model_name="gigachat-20b")
config["dataset"] = {"train_file": "Egor-3926/Dataset_of_Russian_thinking", "val_file": None}

trainer = NPITrainer(config)
trainer.train()
