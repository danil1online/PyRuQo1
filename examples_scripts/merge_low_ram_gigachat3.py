#!/usr/bin/env python3
"""Слияние LoRA GigaChat3 (низкий RAM). npi merge --model gigachat3-10b --low-ram"""
from npi.config import load_config
from npi.merge import LORAMerger
config = load_config(model_name="gigachat3-10b")
config["merge"]["low_ram"] = True
merger = LORAMerger(config)
merger.merge(manage_swap=True)
