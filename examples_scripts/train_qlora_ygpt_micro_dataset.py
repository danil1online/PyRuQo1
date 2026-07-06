#!/usr/bin/env python3
"""Обучение YandexGPT-8B на микро-датасете. pyruqo1 train --model ygpt-5-lite-8b"""
from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer
config = load_config(model_name="ygpt-5-lite-8b")
config["dataset"] = {"train_file": "micro_datasets/university_train.json", "val_file": "micro_datasets/university_val.json"}
trainer = NPITrainer(config)
trainer.train()
