#!/usr/bin/env python3
"""Обучение GigaChat3-10B на большом датасете. npi train --model gigachat3-10b"""
from npi.config import load_config
from npi.training import NPITrainer
trainer = NPITrainer(load_config(model_name="gigachat3-10b"))
trainer.train()
