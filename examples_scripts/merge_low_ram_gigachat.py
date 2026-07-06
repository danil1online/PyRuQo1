#!/usr/bin/env python3
"""Слияние LoRA с низким RAM (< 64 ГБ). npi merge --model gigachat-20b --low-ram"""
from npi.config import load_config
from npi.merge import LORAMerger
config = load_config(model_name="gigachat-20b")
config["merge"]["low_ram"] = True
merger = LORAMerger(config)
merger.merge(manage_swap=True)
