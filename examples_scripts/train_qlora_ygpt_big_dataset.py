#!/usr/bin/env python3
"""Обучение YandexGPT-8B на большом датасете. npi train --model ygpt-5-lite-8b"""
from npi.config import load_config
from npi.training import NPITrainer
trainer = NPITrainer(load_config(model_name="ygpt-5-lite-8b"))
trainer.train()
