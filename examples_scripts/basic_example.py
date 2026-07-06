#!/usr/bin/env python3
"""Базовый пример обучения на готовом датасете HuggingFace. npi train --model gigachat-20b"""
from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer

config = load_config(model_name="gigachat-20b")
config["dataset"] = {"train_file": "Egor-3926/Dataset_of_Russian_thinking", "val_file": None}

trainer = NPITrainer(config)
trainer.train()
