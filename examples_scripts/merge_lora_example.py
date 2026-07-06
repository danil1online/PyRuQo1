#!/usr/bin/env python3
"""Слияние LoRA с базовой моделью. npi merge --model gigachat-20b"""
from npi.config import load_config
from npi.merge import LORAMerger
merger = LORAMerger(load_config(model_name="gigachat-20b"))
merger.merge()
