#!/usr/bin/env python3
"""
Базовый пример обучения на готовом датасете HuggingFace.

Сохранён для обратной совместимости.
Рекомендуемый способ: npi train --model gigachat-20b --train-file <path>
"""

from npi.config import load_config
from npi.training import NPITrainer

config = load_config(model_name="gigachat-20b")
config["dataset"] = {"train_file": "Egor-3926/Dataset_of_Russian_thinking", "val_file": None}

trainer = NPITrainer(config)
trainer.train()
