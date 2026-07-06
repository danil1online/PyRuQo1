#!/usr/bin/env python3
"""Обучение GigaChat-20B на большом датасете. pyruqo1 train --model gigachat-20b"""
from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer
trainer = NPITrainer(load_config(model_name="gigachat-20b"))
trainer.train()
