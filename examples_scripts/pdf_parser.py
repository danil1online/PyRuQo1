#!/usr/bin/env python3
"""Парсинг PDF в чанки. pyruqo1 parse --input ./pdfs"""
from pyruqo1.dataset import PDFParser, TextChunker

parser = PDFParser()
texts = parser.parse_folder("./university_pdfs")

chunker = TextChunker(chunk_size=3500, overlap=500)
chunks = []
for text in texts:
    chunks.extend(chunker.chunk(text))

print(f"Извлечено {len(chunks)} чанков.")
for i, chunk in enumerate(chunks):
    print(f"--- Чанк {i+1} ({len(chunk)} символов) ---")
    print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    print()
