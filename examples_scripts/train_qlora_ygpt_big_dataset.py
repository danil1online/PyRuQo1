#!/usr/bin/env python3
"""Обучение YandexGPT-8B на большом датасете. pyruqo1 train --model ygpt-5-lite-8b"""
from pyruqo1.config import load_config
from pyruqo1.training import NPITrainer
trainer = NPITrainer(load_config(model_name="ygpt-5-lite-8b"))
trainer.train()
