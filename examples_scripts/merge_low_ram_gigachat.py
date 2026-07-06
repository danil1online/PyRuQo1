#!/usr/bin/env python3
"""Слияние LoRA с низким RAM (< 64 ГБ). pyruqo1 merge --model gigachat-20b --low-ram"""
from pyruqo1.config import load_config
from pyruqo1.merge import LORAMerger
config = load_config(model_name="gigachat-20b")
config["merge"]["low_ram"] = True
merger = LORAMerger(config)
merger.merge(manage_swap=True)
