#!/usr/bin/env python3
"""Обучение GigaChat3-10B на микро-датасете. npi train --model gigachat3-10b"""
from npi.config import load_config
from npi.training import NPITrainer
config = load_config(model_name="gigachat3-10b")
config["dataset"] = {"train_file": "micro_datasets/university_train.json", "val_file": "micro_datasets/university_val.json"}
trainer = NPITrainer(config)
trainer.train()
