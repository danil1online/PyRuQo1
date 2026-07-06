#!/usr/bin/env python3
"""Парсинг PDF с формулами (Marker). pyruqo1 parse --input ./pdfs --mode math"""
from pyruqo1.dataset import MathParser, MathChunker

parser = MathParser()
texts = parser.parse_folder("./university_pdfs")

chunker = MathChunker(chunk_size=3500, overlap=500)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

print(f"Извлечено {len(chunks)} чанков с формулами.")
for i, chunk in enumerate(chunks):
    print(f"--- Чанк {i+1} ({len(chunk)} символов) ---")
    print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    print()
