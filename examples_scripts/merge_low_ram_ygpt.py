#!/usr/bin/env python3
"""Слияние LoRA YandexGPT (низкий RAM). npi merge --model ygpt-5-lite-8b --low-ram"""
from npi.config import load_config
from npi.merge import LORAMerger
config = load_config(model_name="ygpt-5-lite-8b")
config["merge"]["low_ram"] = True
merger = LORAMerger(config)
merger.merge(manage_swap=True)
