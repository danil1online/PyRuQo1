#!/usr/bin/env python3
"""Разрезание журналов на статьи. npi split --input ./journals"""
from npi.dataset import JournalSplitter

splitter = JournalSplitter(output_dir="./university_pdfs")
splitter.split_folder("./raw_journals")
