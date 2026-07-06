#!/usr/bin/env python3
"""Слияние LoRA с базовой моделью. pyruqo1 merge --model gigachat-20b"""
from pyruqo1.config import load_config
from pyruqo1.merge import LORAMerger
merger = LORAMerger(load_config(model_name="gigachat-20b"))
merger.merge()
