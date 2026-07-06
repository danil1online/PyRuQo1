#!/usr/bin/env python3
"""Разрезание журналов на статьи. pyruqo1 split --input ./journals"""
from pyruqo1.dataset import JournalSplitter

splitter = JournalSplitter(output_dir="./university_pdfs")
splitter.split_folder("./raw_journals")
