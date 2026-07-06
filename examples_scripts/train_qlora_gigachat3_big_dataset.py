#!/usr/bin/env python3
"""Обучение GigaChat3-10B на большом датасете. pyruqo1 train --model gigachat3-10b"""
from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer
trainer = NPITrainer(load_config(model_name="gigachat3-10b"))
trainer.train()
