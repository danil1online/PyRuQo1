#!/usr/bin/env python3
"""Обучение GigaChat-20B на большом датасете. npi train --model gigachat-20b"""
from npi.config import load_config
from npi.training import NPITrainer
trainer = NPITrainer(load_config(model_name="gigachat-20b"))
trainer.train()
