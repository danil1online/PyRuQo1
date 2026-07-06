#!/usr/bin/env python3
"""Слияние LoRA YandexGPT (низкий RAM). pyruqo1 merge --model ygpt-5-lite-8b --low-ram"""
from pyruqo1.config import load_config
from pyruqo1.merge import LORAMerger
config = load_config(model_name="ygpt-5-lite-8b")
config["merge"]["low_ram"] = True
merger = LORAMerger(config)
merger.merge(manage_swap=True)
