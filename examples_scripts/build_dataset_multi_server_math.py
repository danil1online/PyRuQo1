#!/usr/bin/env python3
"""Генерация математического датасета (Marker + мульти-сервер). pyruqo1 generate --input ./pdfs --mode math"""
from pyruqo1.dataset import MathParser, MathChunker, DatasetGenerator

parser = MathParser()
texts = parser.parse_folder("./university_pdfs")

chunker = MathChunker(chunk_size=3500, overlap=500)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

servers = ["http://192.168.2.52:8079/v1/chat/completions", "http://192.168.2.53:8079/v1/chat/completions"]
generator = DatasetGenerator(servers=servers)
generator.generate_from_chunks(chunks, "university_math_dataset.json", mode="math")
